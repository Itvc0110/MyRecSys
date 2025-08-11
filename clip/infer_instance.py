# infer.py (replace your current script with this cleaned version)

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
import re

from torch.utils.data import DataLoader, TensorDataset
from dcnv3 import DCNv3
from rule_process import get_rulename
from user_process import process_user_data
from item_process import process_clip_item
from processing import process_infer_data  # new centralized prep

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

def _parse_part_index(path):
    m = re.search(r'part_(\d+)\.parquet$', path)
    if m:
        return int(m.group(1))
    # fallback: use enumeration ordering
    return None

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

    # Preprocess user/item if needed (keeps your original flow)
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

    # Load preprocessed small tables (still useful)
    print("\nLoading user and item data...")
    user_df = pd.read_parquet(processed_user_path)
    clip_df = pd.read_parquet(processed_item_path)

    # Build (or reuse) prebuilt cross-join parts
    infer_prep_start = time.time()
    infer_prep = process_infer_data(user_df, clip_df,
                                    num_user=-1, num_clip=-1,
                                    output_dir_path="clip/infer_data",
                                    user_batch_size=10,
                                    force_rebuild=False)  # set True to force rebuild
    infer_files = infer_prep["infer_files"]
    user_profile_path = infer_prep["user_profile_path"]
    print(f"Prepared {len(infer_files)} infer part files in {time.time()-infer_prep_start:.2f} seconds")

    # Load content metadata once
    try:
        content_clip_pl = pl.read_parquet(content_clip_path)
    except FileNotFoundError:
        print(f"Error: Content metadata not found at {content_clip_path}")
        raise

    content_unique = (
        content_clip_pl
        .unique(subset=['content_id'])
        .select(['content_id', 'content_name', 'tag_names', 'type_id'])
    )
    content_dict = {row[0]: (row[1], row[2], row[3]) for row in content_unique.iter_rows()}

    # Load model + warm-up
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = os.path.join(project_root, "model/clip/best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)  # load to CPU first
    expected_input_dim = checkpoint["model_state_dict"]["ECN.dfc.weight"].shape[1]

    model = DCNv3(expected_input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # warm-up single forward pass to initialize CUDA context
    if device.type == 'cuda':
        model.eval()
        with torch.no_grad():
            warm = torch.randn(1, expected_input_dim, device=device)
            _ = model(warm)

    # iterate prebuilt part files and infer
    total_pairs = 0
    num_parts = len(infer_files)
    print(f"\nProcessing {num_parts} parts...")

    for part_idx, part_file in enumerate(sorted(infer_files, key=lambda p: (_parse_part_index(p) or 0))):
        chunk_start = time.time()
        print(f"[Part {part_idx+1}/{num_parts}] {os.path.basename(part_file)}")

        # read part (polars)
        read_start = time.time()
        cross_df = pl.read_parquet(part_file)
        print(f"  ↪︎ Read part: {time.time()-read_start:.2f} seconds")

        # prepare features (keep as polars until to_numpy)
        exclude = {"username", "content_id", "profile_id"}
        to_convert = [c for c in cross_df.columns if c not in exclude]
        if not to_convert:
            print("  ↪︎ No feature columns found in part, skipping.")
            continue

        # cast numeric features to float32
        cast_map = {col: pl.Float32 for col in to_convert}
        cross_df = cross_df.cast(cast_map)

        # extract ids and features
        interaction_df = cross_df.select(["username", "content_id", "profile_id"]).to_pandas()
        features_np = cross_df.select(to_convert).to_numpy()

        # Inference
        infer_start = time.time()
        infer_tensor = torch.tensor(features_np, dtype=torch.float32)
        infer_loader = DataLoader(TensorDataset(infer_tensor), batch_size=2048, shuffle=False)
        predictions = infer(model, infer_loader, device)
        total_pairs += len(predictions)
        print(f"  ↪︎ Inference: {time.time()-infer_start:.2f} seconds")

        # Collect top N per profile in this part
        collect_start = time.time()
        temp_dict = {}
        for pid, user, cid, score in zip(
                interaction_df['profile_id'],
                interaction_df['username'],
                interaction_df['content_id'],
                predictions
        ):
            temp_dict.setdefault(pid, []).append((cid, float(score)))

        chunk_result = {}
        # create mapping username quickly from user_profile_data if needed
        # but we have username in interaction_df; we will reuse user_profile_df from first row
        for pid, items in temp_dict.items():
            sorted_items = sorted(items, key=lambda x: x[1], reverse=True)[:TOP_N]
            # username - try to get from the interaction_df (first occurrence)
            username = interaction_df.loc[interaction_df['profile_id'] == pid, 'username'].iloc[0]
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
                    'username': username,
                    'profile_id': pid
                }
            }
        print(f"  ↪︎ Collect top {TOP_N}: {time.time()-collect_start:.2f} seconds")

        # Add metadata
        meta_start = time.time()
        for pid, pdata in chunk_result.items():
            for cid, cdata in pdata['suggested_content'].items():
                meta = content_dict.get(cid)
                if meta:
                    cdata['content_name'], cdata['tag_names'], cdata['type_id'] = map(str, meta)
        print(f"  ↪︎ Add metadata: {time.time()-meta_start:.2f} seconds")

        # Rank & rule assignment
        rank_start = time.time()
        reordered_chunk = rank_result(chunk_result, TOP_N)
        result_with_rule = get_rulename(reordered_chunk, rule_info_path, tags_path)
        print(f"  ↪︎ Ranking & rule assignment: {time.time()-rank_start:.2f} seconds")

        # Save per-chunk outputs
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

        base_idx = _parse_part_index(part_file)
        out_idx = base_idx if base_idx is not None else (part_idx + 1)
        with open(project_root / f"clip/result/result_chunk_{out_idx}.json", 'w', encoding='utf-8') as f:
            json.dump(result_with_rule, f, indent=4, ensure_ascii=False)
        with open(project_root / f"clip/result/rulename_chunk_{out_idx}.json", 'w', encoding='utf-8') as f:
            json.dump(homepage_rule, f, indent=4, ensure_ascii=False)
        with open(project_root / f"clip/result/rule_content_chunk_{out_idx}.txt", "w", encoding='utf-8') as f:
            f.write(rule_content)
        print(f"  ↪︎ Saved chunk files in {time.time()-save_start:.2f} seconds")

        # Cleanup aggressively
        del predictions, temp_dict, chunk_result, cross_df, features_np, infer_tensor, infer_loader
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  ↪︎ Total part time: {time.time()-chunk_start:.2f} seconds\n")

    # Merge per-chunk files (post-step)
    merge_start = time.time()
    final_result = {}
    final_rulename = []
    final_rule_content = ""
    chunk_files = sorted(glob.glob(str(project_root / "clip" / "result" / "result_chunk_*.json")))
    for fpath in chunk_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            final_result.update(json.load(f))
    rulename_files = sorted(glob.glob(str(project_root / "clip" / "result" / "rulename_chunk_*.json")))
    for fpath in rulename_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            final_rulename.extend(json.load(f))
    rule_content_files = sorted(glob.glob(str(project_root / "clip" / "result" / "rule_content_chunk_*.txt")))
    for fpath in rule_content_files:
        with open(fpath, 'r', encoding='utf-8') as f:
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
