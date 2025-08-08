import os
import json
import time
import gc
import glob
from pathlib import Path
from math import ceil
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dcnv3 import DCNv3
from rule_process import get_rulename
from user_process import process_user_data
from item_process import process_clip_item

def rank_result(data, n):
    reordered_data = {}
    for user_id in data:
        user_scores = []
        for content_id, content in data[user_id]['suggested_content'].items():
            user_scores.append({
                'content_id': content_id,
                'content_name': content['content_name'],
                'tag_names': content['tag_names'],
                'type_id': content['type_id'],
                'score': content['score']
            })
        sorted_user_scores = sorted(user_scores, key=lambda x: x['score'], reverse=True)[:n]
        reordered_data[user_id] = {
            'suggested_content': {
                film['content_id']: {
                    'content_name': film['content_name'],
                    'tag_names': film['tag_names'],
                    'type_id': film['type_id'],
                    'score': film['score']
                } for film in sorted_user_scores
            },
            'user': data[user_id]['user']
        }
    return reordered_data

def infer(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Inference"):
            inputs = batch[0].to(device)
            outputs = model(inputs)
            predictions.extend(outputs['y_pred'].detach().cpu().numpy())
        torch.cuda.empty_cache()
    return predictions

if __name__ == "__main__":
    start_time = time.time()
    TOP_N = 200

    project_root = Path().resolve()
    os.makedirs(project_root / "clip" / "result", exist_ok=True)
    os.makedirs(project_root / "clip" / "infer_data", exist_ok=True)

    # File paths
    user_data_path = os.path.join(project_root, "month_mytv_info.parquet")
    clip_data_path = os.path.join(project_root, "mytv_vmp_content")
    processed_user_path = os.path.join(project_root, "clip/infer_data/user_data.parquet")
    processed_item_path = os.path.join(project_root, "clip/infer_data/clip_item_data.parquet")
    content_clip_path = os.path.join(project_root, "clip/infer_data/merged_content_clips.parquet")
    tags_path = os.path.join(project_root, "tags")
    rule_info_path = os.path.join(project_root, "rule_info.parquet")
    result_json_path = os.path.join(project_root, "clip/result/result.json")
    rulename_json_path = os.path.join(project_root, "clip/result/rulename.json")
    rule_content_path = os.path.join(project_root, "clip/result/rule_content.txt")

    # Preprocess
    print("Preprocessing user and item data...")
    preprocess_start = time.time()
    try:
        if not os.path.exists(processed_user_path):
            print("  ↪︎ Processing user data...")
            process_user_data(user_data_path, output_dir="clip/infer_data", num_user=-1, mode='infer')
        else:
            print("  ↪︎ User data already exists, skipping preprocessing.")
        if not os.path.exists(processed_item_path):
            print("  ↪︎ Processing item data...")
            process_clip_item(clip_data_path, output_dir="clip/infer_data", num_clip=-1, mode='infer')
        else:
            print("  ↪︎ Item data already exists, skipping preprocessing.")
    except FileNotFoundError as e:
        print(f"Error during preprocessing: {e}")
        exit(1)
    print(f"Preprocessing completed in {time.time()-preprocess_start:.2f} seconds")


    # Load data
    print("\nLoading user and item data...")
    try:
        user_df = pd.read_parquet(processed_user_path)
        clip_df = pd.read_parquet(processed_item_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    # Duration mapping -> build user_profile_df
    duration_dir = os.path.join(project_root, "clip/merged_duration")
    if not os.path.exists(duration_dir):
        print(f"Error: Duration directory not found at {duration_dir}")
        print("Run merge_parquet_files from duration_process.py to generate duration files.")
        exit(1)
    durations = glob.glob(os.path.join(duration_dir, "*.parquet"))
    if not durations:
        print(f"Error: No duration data found in {duration_dir}")
        exit(1)
    user_profile_list = []
    for duration in durations:
        df = pd.read_parquet(duration, columns=["username", "profile_id"])
        user_profile_list.append(df.drop_duplicates())
    user_profile_df = pd.concat(user_profile_list, ignore_index=True).drop_duplicates()
    user_profile_df = user_profile_df.merge(user_df, on="username", how="inner")
    print(f"Data loaded in {time.time()-preprocess_start:.2f} seconds")


    # Load model once
    checkpoint_path = os.path.join(project_root, "model/clip/best_model.pth")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        exit(1)
    expected_input_dim = checkpoint["model_state_dict"]["ECN.dfc.weight"].shape[1]
    model = DCNv3(expected_input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])


    # Preload clip and content metadata once
    clip_pl = pl.from_pandas(clip_df)
    total_contents = clip_pl.height
    print(f"\nTotal unique contents: {total_contents:,}")

    try:
        content_clip_pl = pl.read_parquet(content_clip_path)
    except FileNotFoundError:
        print(f"Error: Content metadata not found at {content_clip_path}")
        exit(1)

    content_unique = (
        content_clip_pl
        .unique(subset=['content_id'])
        .select(['content_id', 'content_name', 'tag_names', 'type_id'])
    )
    # content_dict: content_id -> (name, tag_names, type_id)
    content_dict = {row[0]: (row[1], row[2], row[3]) for row in content_unique.iter_rows()}


    # Prepare pipeline utilities
    profile_to_username = dict(zip(user_profile_df['profile_id'], user_profile_df['username']))

    # chunk control
    user_batch_size = 30  
    num_users = len(user_profile_df)
    estimated_num_chunks = ceil(num_users / user_batch_size)
    print(f"\nProcessing {num_users} users in ~{estimated_num_chunks} chunks (batch size {user_batch_size})...")

    # Thread pool for async saving (ranking+rule+file write)
    executor = ThreadPoolExecutor(max_workers=2)

    # We'll track the real number of produced chunk files (in case fallback per-user processing expands it)
    produced_chunks = 0
    total_pairs = 0
    
    def process_and_save_chunk(chunk_idx, predictions, interaction_df):
        """Process predictions (list/array) and interaction_df (pandas) and save chunk files."""
        collect_start = time.time()

        # build pid -> list[(cid, score)]
        temp_dict = {}
        for pid, cid, score in zip(interaction_df['profile_id'],
                                   interaction_df['content_id'],
                                   predictions):
            temp_dict.setdefault(pid, []).append((cid, float(score)))

        # Build chunk_result structure
        chunk_result = {}
        for pid, items in temp_dict.items():
            sorted_items = sorted(items, key=lambda x: x[1], reverse=True)[:TOP_N]
            chunk_result[pid] = {
                'suggested_content': {
                    cid: {
                        'content_name': '',
                        'tag_names': '',
                        'type_id': '',
                        'score': score
                    } for cid, score in sorted_items
                },
                'user': {
                    'username': profile_to_username.get(pid, ""),
                    'profile_id': pid
                }
            }

        print(f"[Chunk {chunk_idx}] ↪︎ Collect top {TOP_N}: {time.time()-collect_start:.2f} seconds")

        # Add content metadata
        for pid, pdata in chunk_result.items():
            for cid, cdata in pdata['suggested_content'].items():
                meta = content_dict.get(cid)
                if meta:
                    cdata['content_name'], cdata['tag_names'], cdata['type_id'] = map(str, meta)

        # Rank + rule assignment
        rank_start = time.time()
        reordered_chunk = rank_result(chunk_result, TOP_N)
        result_with_rule = get_rulename(reordered_chunk, rule_info_path, tags_path)
        print(f"[Chunk {chunk_idx}] ↪︎ Ranking & rule assignment: {time.time()-rank_start:.2f} seconds")

        # Save per-chunk
        save_start = time.time()
        homepage_rule = []
        rule_content = ""
        for pid, info in result_with_rule.items():
            rulename_json_file = {'pid': pid, 'd': {}}
            rule_content += str(pid) + '|'
            for key, content in info['suggested_content'].items():
                if content['rule_id'] != -100:
                    rulename_json_file['d'].setdefault(str(content['rule_id']), {})[key] = content['type_id']
            for rule, content_ids in rulename_json_file['d'].items():
                rule_content += str(rule) + ''.join(f'${v}#{k}' for k, v in content_ids.items()) + ';'
            rule_content = rule_content.rstrip(';') + '\n'
            # convert dict to comma-joined string as before
            rulename_json_file['d'] = ','.join(rulename_json_file['d'])
            homepage_rule.append(rulename_json_file)

        # write files
        result_chunk_path = project_root / f"clip/result/result_chunk_{chunk_idx}.json"
        rulename_chunk_path = project_root / f"clip/result/rulename_chunk_{chunk_idx}.json"
        rule_content_chunk_path = project_root / f"clip/result/rule_content_chunk_{chunk_idx}.txt"

        with open(result_chunk_path, 'w', encoding='utf-8') as f:
            json.dump(result_with_rule, f, indent=4, ensure_ascii=False)
        with open(rulename_chunk_path, 'w', encoding='utf-8') as f:
            json.dump(homepage_rule, f, indent=4, ensure_ascii=False)
        with open(rule_content_chunk_path, 'w', encoding='utf-8') as f:
            f.write(rule_content)

        print(f"[Chunk {chunk_idx}] ↪︎ Saved chunk files in {time.time()-save_start:.2f} seconds")

        # cleanup worker-local large objects
        del predictions, interaction_df, chunk_result, reordered_chunk, result_with_rule
        gc.collect()

    # Main pipelined loop
    prev_future = None
    chunk_id_counter = 0  # actual chunk index for filenames (1-based)
    try:
        for i in range(0, num_users, user_batch_size):
            # process batch of users user_profile_df[i : i+user_batch_size]
            chunk_start = time.time()
            user_chunk = user_profile_df.iloc[i:i+user_batch_size]
            print(f"[Batch users {i+1}-{i+len(user_chunk)}] Processing...")

            # Attempt a cross join for the mini-batch
            cross_start = time.time()
            try:
                user_chunk_pl = pl.from_pandas(user_chunk)
                cross_df = user_chunk_pl.join(clip_pl, how="cross")
                # Polars numeric cast here (stay in Polars)
                exclude = {"username", "content_id", "profile_id"}
                to_convert = [c for c in cross_df.columns if c not in exclude]
                if to_convert:
                    # cast to float32 (columns that are convertible)
                    cast_map = {col: pl.Float32 for col in to_convert}
                    cross_df = cross_df.cast(cast_map)

                # Prepare small objects for inference (convert at last moment)
                interaction_df = cross_df.select(["username", "content_id", "profile_id"]).to_pandas()
                features_np = cross_df.select(to_convert).to_numpy() if to_convert else np.zeros((len(interaction_df), 0), dtype="float32")
                cross_time = time.time() - cross_start
                print(f"  ↪︎ Cross join: {cross_time:.2f} seconds")
            except Exception as e:
                # Fallback: cross join failed (likely memory). Process per-user in this batch sequentially.
                print(f"  ↪︎ Cross join failed for batch ({i+1}-{i+len(user_chunk)}): {e}")
                print("  ↪︎ Falling back to per-user processing for this batch (slower but safe).")
                # process each user individually
                for single_idx in range(len(user_chunk)):
                    single_user = user_chunk.iloc[[single_idx]]
                    try:
                        su_pl = pl.from_pandas(single_user)
                        single_cross = su_pl.join(clip_pl, how="cross")
                        exclude = {"username", "content_id", "profile_id"}
                        to_convert = [c for c in single_cross.columns if c not in exclude]
                        if to_convert:
                            single_cross = single_cross.cast({col: pl.Float32 for col in to_convert})
                        interaction_df = single_cross.select(["username", "content_id", "profile_id"]).to_pandas()
                        features_np = single_cross.select(to_convert).to_numpy() if to_convert else np.zeros((len(interaction_df), 0), dtype="float32")
                    except Exception as e2:
                        print(f"    ↪︎ Per-user cross join failed for user index {i+single_idx}: {e2}")
                        # skip this user if even single-user join fails
                        continue

                    # inference for single-user
                    infer_start = time.time()
                    if features_np.size > 0:
                        infer_tensor = torch.tensor(features_np, dtype=torch.float32)
                        infer_loader = DataLoader(TensorDataset(infer_tensor), batch_size=32768, shuffle=False)
                        predictions = infer(model, infer_loader, device)
                    else:
                        predictions = []
                    total_pairs += len(predictions)
                    print(f"    ↪︎ Inference (single user): {time.time()-infer_start:.2f} seconds")

                    # Schedule save for this single-user as a chunk
                    chunk_id_counter += 1
                    if prev_future:
                        prev_future.result()
                    prev_future = executor.submit(process_and_save_chunk, chunk_id_counter, predictions, interaction_df)

                    # cleanup single-user intermediates
                    del predictions, interaction_df, features_np
                    torch.cuda.empty_cache()
                    gc.collect()

                # after finishing per-user fallback, continue to next batch
                continue

            # Inference for the successful batch cross-join
            infer_start = time.time()
            if features_np.size > 0:
                infer_tensor = torch.tensor(features_np, dtype=torch.float32)
                infer_loader = DataLoader(TensorDataset(infer_tensor), batch_size=32768, shuffle=False)
                predictions = infer(model, infer_loader, device)
            else:
                predictions = []
            total_pairs += len(predictions)
            print(f"  ↪︎ Inference: {time.time()-infer_start:.2f} seconds")

            # Kick off save for this batch (as one chunk)
            chunk_id_counter += 1
            if prev_future:
                # wait for previous save to finish to avoid unbounded memory growth in writer
                prev_future.result()
            prev_future = executor.submit(process_and_save_chunk, chunk_id_counter, predictions, interaction_df)

            # cleanup per-batch intermediates
            del predictions, interaction_df, features_np, cross_df, user_chunk, user_chunk_pl
            torch.cuda.empty_cache()
            gc.collect()

            print(f"  ↪︎ Total batch time (prep+infer): {time.time()-chunk_start:.2f} seconds\n")

    finally:
        # Ensure last save finishes and executor is shutdown
        if prev_future:
            prev_future.result()
        executor.shutdown(wait=True)

    produced_chunks = chunk_id_counter
    print(f"\nProduced {produced_chunks} chunk files. Starting post-step merge...")

    # Post-step: Merge per-chunk files into final outputs
    merge_start = time.time()
    final_result = {}
    final_rulename = []
    final_rule_content = ""

    for idx in range(1, produced_chunks + 1):
        result_path = project_root / f"clip/result/result_chunk_{idx}.json"
        rulename_path = project_root / f"clip/result/rulename_chunk_{idx}.json"
        rule_content_path_chunk = project_root / f"clip/result/rule_content_chunk_{idx}.txt"

        if result_path.exists():
            with open(result_path, 'r', encoding='utf-8') as f:
                final_result.update(json.load(f))
        if rulename_path.exists():
            with open(rulename_path, 'r', encoding='utf-8') as f:
                final_rulename.extend(json.load(f))
        if rule_content_path_chunk.exists():
            with open(rule_content_path_chunk, 'r', encoding='utf-8') as f:
                final_rule_content += f.read()

    # Write final merged outputs
    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
    with open(rulename_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_rulename, f, indent=4, ensure_ascii=False)
    with open(project_root / "clip/result/rule_content.txt", 'w', encoding='utf-8') as f:
        f.write(final_rule_content)

    print(f"Merged final files in {time.time()-merge_start:.2f} seconds")

    # Final summary
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Total user-item pairs inferred: {total_pairs:,}")
    if total_pairs > 0:
        print(f"Average inference time per pair: {total_time / total_pairs:.6f} seconds")
    else:
        print("No pairs were inferred.")