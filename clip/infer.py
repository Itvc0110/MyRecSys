import os
import pandas as pd
import polars as pl
import torch
from tqdm import tqdm
import json
import time
from pathlib import Path
from glob import glob

from torch.utils.data import DataLoader, TensorDataset
from processing import process_infer_data
from dcnv3 import DCNv3
from rule_process import get_rulename
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


# --- Replace main block (if __name__ == "__main__") with this ---

if __name__ == "__main__":
    start_time = time.time()
    TOP_N = 200
    project_root = Path().resolve()

    # Prepare output folders
    os.makedirs(project_root / "clip" / "result", exist_ok=True)
    os.makedirs(project_root / "clip" / "infer_data", exist_ok=True)

    # Paths
    user_data_path = os.path.join(project_root, "month_mytv_info.parquet")
    clip_data_path = os.path.join(project_root, "mytv_vmp_content")
    content_clip_path = os.path.join(project_root, "clip/infer_data/merged_content_clips.parquet")
    tags_path = os.path.join(project_root, "tags")
    rule_info_path = os.path.join(project_root, "rule_info.parquet")

    result_json_path = os.path.join(project_root, "clip/result/result.json")
    rulename_json_path = os.path.join(project_root, "clip/result/rulename.json")
    rule_content_path = os.path.join(project_root, "clip/result/rule_content.txt")

    # Load model
    checkpoint_path = "model/clip/best_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    expected_input_dim = checkpoint["model_state_dict"]["ECN.dfc.weight"].shape[1]
    model = DCNv3(expected_input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    result_dict = {}
    total_pairs = 0

    # Optionally load content metadata early for mapping after inference
    # (this uses the merged content file saved by process_clip_item)
    try:
        content_clip_pl = pl.read_parquet(content_clip_path)
        content_unique = (
            content_clip_pl
            .unique(subset=['content_id'])
            .select(['content_id', 'content_name', 'tag_names', 'type_id'])
        )
        content_dict = {row[0]: (row[1], row[2], row[3]) for row in content_unique.iter_rows()}
        print(f"[Info] Loaded {len(content_dict)} unique content metadata entries from {content_clip_path}")
    except Exception as e:
        print(f"[Warning] Could not load content metadata ({content_clip_path}): {e}")
        content_dict = {}

    # --- On-the-fly pipeline ---
    for chunk_idx, user_chunk, clip_df in process_infer_data(user_data_path, clip_data_path,
                                                              num_user=-1, num_clip=-1,
                                                              output_dir_path="clip/infer_data",
                                                              user_batch_size=10):

        chunk_start = time.time()
        print(f"\n[Chunk {chunk_idx}] Processing {len(user_chunk)} users...")

        # Cross join (polars) -> pandas
        t0 = time.time()
        cross_pl = pl.from_pandas(user_chunk).join(pl.from_pandas(clip_df), how="cross")
        cross_df = cross_pl.fill_null(0).to_pandas()
        cross_elapsed = time.time() - t0
        print(f"  ↪︎ Cross join: {len(cross_df)} rows in {cross_elapsed:.2f}s")

        # Prepare features (drop identifiers)
        exclude = {'username', 'content_id', 'profile_id'}
        to_convert = [col for col in cross_df.columns if col not in exclude]
        cross_df[to_convert] = cross_df[to_convert].apply(pd.to_numeric, errors='coerce')
        cross_df = cross_df.dropna().reset_index(drop=True)
        cross_df = cross_df.astype({col: 'float32' for col in to_convert})

        if cross_df.shape[0] == 0:
            print(f"  ↪︎ Skipping: cross chunk produced 0 rows after cleaning")
            continue

        interaction_df = cross_df[['username', 'content_id', 'profile_id']]
        features = cross_df.drop(columns=['username', 'content_id', 'profile_id'])

        # Check feature dimension matches model expectation
        if features.shape[1] != expected_input_dim:
            raise RuntimeError(
                f"Feature dimension mismatch: features.shape[1]={features.shape[1]} "
                f"but model expected {expected_input_dim}. "
                f"Columns: {list(features.columns)}"
            )

        # Inference
        t0 = time.time()
        infer_tensor = torch.tensor(features.to_numpy(), dtype=torch.float32)
        # DataLoader accepts datasets of tensors; we keep single-tensor dataset
        infer_loader = DataLoader(TensorDataset(infer_tensor), batch_size=2048, shuffle=False,
                                  pin_memory=(device.type == 'cuda'))
        predictions = infer(model, infer_loader, device)
        infer_elapsed = time.time() - t0
        print(f"  ↪︎ Inference took {infer_elapsed:.2f}s for {len(predictions):,} pairs")

        total_pairs += len(predictions)

        # Store results into result_dict
        for pid, user, cid, score in zip(
            interaction_df['profile_id'],
            interaction_df['username'],
            interaction_df['content_id'],
            predictions
        ):
            pid = str(pid)
            cid = str(cid)
            result_dict.setdefault(pid, {})
            result_dict[pid].setdefault('suggested_content', {})
            result_dict[pid]['suggested_content'][cid] = {
                'content_name': '',  # will fill from metadata below
                'tag_names': '',
                'type_id': '',
                'score': float(score)
            }
            result_dict[pid]['user'] = {'username': user, 'profile_id': pid}

        # Cleanup to free memory
        del cross_pl, cross_df, interaction_df, features, infer_tensor, infer_loader, predictions
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        print(f"[Chunk {chunk_idx}] Done in {time.time() - chunk_start:.2f}s | Cumulative pairs: {total_pairs:,}")

    # Add content metadata (content_name, tag_names, type_id) into result_dict
    if content_dict:
        for pid, pdata in result_dict.items():
            for cid, cdata in pdata['suggested_content'].items():
                meta = content_dict.get(str(cid))
                if meta:
                    cdata['content_name'], cdata['tag_names'], cdata['type_id'] = map(str, meta)

    # Ranking & rule assignment
    print("\nRanking results...")
    ranked = rank_result(result_dict, TOP_N)

    print("\nApplying rules...")
    ruled = get_rulename(ranked, rule_info_path, tags_path)

    # Build homepage rules / rule_content
    homepage_rule = []
    rule_content = ""
    for pid, info in ruled.items():
        rulename_json_file = {'pid': pid, 'd': {}}
        rule_content += str(pid) + '|'
        for cid, content in info['suggested_content'].items():
            if content.get('rule_id', -100) != -100:
                rulename_json_file['d'].setdefault(str(content['rule_id']), {})[cid] = content['type_id']
        for rule, content_ids in rulename_json_file['d'].items():
            rule_content += str(rule) + ',' + ''.join(f'${v}#{k}' for k, v in content_ids.items()) + ';'
        rule_content = rule_content.rstrip(';') + '\n'
        rulename_json_file['d'] = ','.join(rulename_json_file['d'])
        homepage_rule.append(rulename_json_file)

    # Save outputs
    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(ruled, f, indent=4, ensure_ascii=False)
    with open(rulename_json_path, 'w', encoding='utf-8') as f:
        json.dump(homepage_rule, f, indent=4, ensure_ascii=False)
    with open(rule_content_path, 'w', encoding='utf-8') as f:
        f.write(rule_content)

    elapsed = time.time() - start_time
    avg_per_pair = (elapsed / total_pairs) if total_pairs else float('nan')
    print(f"\n✅ Inference pipeline completed in {elapsed:.2f}s")
    print(f"Total user-item pairs: {total_pairs:,}")
    print(f"Average time per pair: {avg_per_pair:.6f} seconds")
