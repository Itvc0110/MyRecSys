import os
import pandas as pd
import polars as pl
import gc
import torch
from tqdm import tqdm
import json
import time
from pathlib import Path
import glob
from math import ceil

from torch.utils.data import DataLoader, TensorDataset
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

def infer(model, data, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(data, desc="Inference"):
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
    print(f"Preprocessing completed in {time.time()-preprocess_start:.2f} seconds")

    # Load data
    print("\nLoading user and item data...")
    user_df = pd.read_parquet(processed_user_path)
    clip_df = pd.read_parquet(processed_item_path)

    # Duration mapping
    duration_dir = os.path.join(project_root, "clip/merged_duration")
    durations = glob.glob(os.path.join(duration_dir, "*.parquet"))
    user_profile_list = []
    for duration in durations:
        df = pd.read_parquet(duration, columns=["username", "profile_id"])
        user_profile_list.append(df.drop_duplicates())
    user_profile_df = pd.concat(user_profile_list, ignore_index=True).drop_duplicates()
    user_profile_df = user_profile_df.merge(user_df, on="username", how="inner")
    print(f"Data loaded in {time.time()-preprocess_start:.2f} seconds")

    # Load model once
    checkpoint_path = os.path.join(project_root, "model/clip/best_model.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    expected_input_dim = checkpoint["model_state_dict"]["ECN.dfc.weight"].shape[1]
    model = DCNv3(expected_input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load clip_pl once
    clip_pl = pl.from_pandas(clip_df)

    # Print total contents
    total_contents = clip_pl.height
    print(f"\nTotal unique contents: {total_contents:,}")

    # Load content metadata
    content_clip_pl = pl.read_parquet(content_clip_path)
    content_unique = (
        content_clip_pl
        .unique(subset=['content_id'])
        .select(['content_id', 'content_name', 'tag_names', 'type_id'])
    )
    content_dict = {row[0]: (row[1], row[2], row[3]) for row in content_unique.iter_rows()}

    # Process in chunks
    user_batch_size = 10
    num_users = len(user_profile_df)
    num_chunks = ceil(num_users / user_batch_size)
    total_pairs = 0

    print(f"\nProcessing {num_users} users in {num_chunks} chunks...")
    for i in range(num_chunks):
        chunk_start = time.time()
        print(f"[Chunk {i+1}/{num_chunks}] Processing...")

        # User chunk
        start_idx = i * user_batch_size
        end_idx = min((i+1) * user_batch_size, num_users)
        user_chunk = user_profile_df.iloc[start_idx:end_idx]

        # Cross join in Polars
        cross_start = time.time()
        user_chunk_pl = pl.from_pandas(user_chunk)
        cross_df = user_chunk_pl.join(clip_pl, how="cross")

        # Separate features & IDs in Polars
        exclude = {"username", "content_id", "profile_id"}
        to_convert = [col for col in cross_df.columns if col not in exclude]
        cross_df = cross_df.cast({col: pl.Float32 for col in to_convert})

        interaction_df = cross_df.select(["username", "content_id", "profile_id"]).to_pandas()
        features_np = cross_df.select(to_convert).to_numpy()
        print(f"  ↪︎ Cross join: {time.time()-cross_start:.2f} seconds")

        # Inference
        infer_start = time.time()
        infer_tensor = torch.tensor(features_np, dtype=torch.float32)
        infer_loader = DataLoader(TensorDataset(infer_tensor), batch_size=2048, shuffle=False)
        predictions = infer(model, infer_loader, device)
        total_pairs += len(predictions)
        print(f"  ↪︎ Inference: {time.time()-infer_start:.2f} seconds")

        # Collect top N
        collect_start = time.time()
        chunk_result = {}
        temp_dict = {}
        for pid, user, cid, score in zip(
            interaction_df['profile_id'],
            interaction_df['username'],
            interaction_df['content_id'],
            predictions
        ):
            temp_dict.setdefault(pid, []).append((cid, float(score)))
        for pid in temp_dict:
            sorted_items = sorted(temp_dict[pid], key=lambda x: x[1], reverse=True)[:TOP_N]
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
                    'username': user_profile_df.loc[user_profile_df['profile_id'] == pid, 'username'].iloc[0],
                    'profile_id': pid
                }
            }
        print(f"  ↪︎ Collect top {TOP_N}: {time.time()-collect_start:.2f} seconds")

        # Add metadata
        for pid, pdata in chunk_result.items():
            for cid, cdata in pdata['suggested_content'].items():
                meta = content_dict.get(cid)
                if meta:
                    cdata['content_name'], cdata['tag_names'], cdata['type_id'] = map(str, meta)

        # Rank + rule
        rank_start = time.time()
        reordered_chunk = rank_result(chunk_result, TOP_N)
        result_with_rule = get_rulename(reordered_chunk, rule_info_path, tags_path)
        print(f"  ↪︎ Ranking & rule assignment: {time.time()-rank_start:.2f} seconds")

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
            rulename_json_file['d'] = ','.join(rulename_json_file['d'])
            homepage_rule.append(rulename_json_file)

        with open(project_root / f"clip/result/result_chunk_{i+1}.json", 'w', encoding='utf-8') as f:
            json.dump(result_with_rule, f, indent=4, ensure_ascii=False)
        with open(project_root / f"clip/result/rulename_chunk_{i+1}.json", 'w', encoding='utf-8') as f:
            json.dump(homepage_rule, f, indent=4, ensure_ascii=False)
        with open(project_root / f"clip/result/rule_content_chunk_{i+1}.txt", "w", encoding='utf-8') as f:
            f.write(rule_content)
        print(f"  ↪︎ Saved chunk files in {time.time()-save_start:.2f} seconds")

        # Cleanup aggressively
        del predictions, temp_dict, chunk_result, cross_df, features_np, infer_tensor, infer_loader
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  ↪︎ Total chunk time: {time.time()-chunk_start:.2f} seconds\n")

    # Merge per-chunk files
    merge_start = time.time()
    final_result = {}
    final_rulename = []
    final_rule_content = ""
    for i in range(num_chunks):
        with open(project_root / f"clip/result/result_chunk_{i+1}.json", 'r', encoding='utf-8') as f:
            final_result.update(json.load(f))
        with open(project_root / f"clip/result/rulename_chunk_{i+1}.json", 'r', encoding='utf-8') as f:
            final_rulename.extend(json.load(f))
        with open(project_root / f"clip/result/rule_content_chunk_{i+1}.txt", 'r', encoding='utf-8') as f:
            final_rule_content += f.read()

    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
    with open(rulename_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_rulename, f, indent=4, ensure_ascii=False)
    with open(rule_content_path, "w", encoding='utf-8') as f:
        f.write(final_rule_content)
    print(f"Merged final files in {time.time()-merge_start:.2f} seconds")

    # Summary
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Total user-item pairs: {total_pairs:,}")
    print(f"Average inference time per pair: {total_time / total_pairs:.6f} seconds")
