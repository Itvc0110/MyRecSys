import os
import pandas as pd
import polars as pl
import torch
from tqdm import tqdm
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from user_process import process_user_data
from item_process import process_clip_item
from dcnv3 import DCNv3
from rule_process import get_rulename_parallel

# -------------------------
# Utility: Parallel ranking
# -------------------------
def _rank_user(args):
    user_id, suggested_content, top_n = args
    user_scores = sorted(
        suggested_content.items(),
        key=lambda kv: kv[1]['score'],
        reverse=True
    )[:top_n]
    return user_id, dict(user_scores)

def rank_result_parallel(data, n, max_workers=None, chunk_size=5000):
    items = list(data.items())
    chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
    
    def _process_chunk(chunk):
        out = {}
        for uid, user_data in chunk:
            scores = sorted(
                user_data['suggested_content'].items(),
                key=lambda kv: kv[1]['score'],
                reverse=True
            )[:n]
            out[uid] = {'suggested_content': dict(scores), 'user': user_data['user']}
        return out

    with ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as ex:
        results = list(ex.map(_process_chunk, chunks))
    
    merged = {}
    for r in results:
        merged.update(r)
    return merged

# -------------------------
# Torch inference function
# -------------------------
def infer(model, features, batch_size, device):
    model.eval()
    predictions = []
    tensor = torch.tensor(features, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(tensor, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            inputs = batch.to(device)
            outputs = model(inputs)
            predictions.extend(outputs['y_pred'].detach().cpu().numpy())
            torch.cuda.empty_cache()
    return predictions

# -------------------------
# Streaming inference pipeline
# -------------------------
if __name__ == "__main__":
    start_time = time.time()
    TOP_N = 200
    USER_BATCH_SIZE = 50  # tune for memory

    project_root = Path().resolve()
    os.makedirs(project_root / "clip" / "result", exist_ok=True)

    # Input paths
    user_data_path = os.path.join(project_root, "month_mytv_info.parquet")
    clip_data_path = os.path.join(project_root, "mytv_vmp_content")
    content_clip_path = os.path.join(project_root, "clip/infer_data/merged_content_clips.parquet")
    tags_path = os.path.join(project_root, "tags")
    rule_info_path = os.path.join(project_root, "rule_info.parquet")

    # Output paths
    result_json_path = os.path.join(project_root, "clip/result/result.json")
    rulename_json_path = os.path.join(project_root, "clip/result/rulename.json")
    rule_content_path = os.path.join(project_root, "clip/result/rule_content.txt")

    # Load data
    print("Loading user & clip data...")
    user_df = process_user_data(user_data_path, "clip/infer_data", num_user=-1, mode='infer')
    clip_df = process_clip_item(clip_data_path, "clip/infer_data", num_clip=-1, mode='infer')
    clip_df['content_id'] = clip_df['content_id'].astype(str)
    total_users = len(user_df)
    total_clips = len(clip_df)
    print(f"Loaded {total_users} users × {total_clips} clips = {total_users * total_clips:,} pairs (streamed)")

    # Load model
    checkpoint_path = "model/clip/best_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    expected_input_dim = checkpoint["model_state_dict"]["ECN.dfc.weight"].shape[1]
    model = DCNv3(expected_input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Prepare clip data in Polars
    clip_pl = pl.from_pandas(clip_df)

    result_dict = {}
    total_pairs = 0

    # Stream user batches
    for start in range(0, len(user_df), USER_BATCH_SIZE):
        batch_start = time.time()
        user_chunk_pd = user_df.iloc[start:start + USER_BATCH_SIZE]
        user_chunk_pl = pl.from_pandas(user_chunk_pd)

        # Cross join in memory
        cross_chunk = user_chunk_pl.join(clip_pl, how="cross")

        # Convert to tensors
        exclude = {'username', 'content_id', 'profile_id'}
        interaction_df = cross_chunk.select(['username', 'content_id', 'profile_id']).to_pandas()
        features = cross_chunk.drop(list(exclude)).to_numpy(dtype='float32')

        # Inference
        preds = infer(model, features, batch_size=2048, device=device)
        total_pairs += len(preds)
        print(f"[Batch {start//USER_BATCH_SIZE+1}] {len(preds):,} pairs processed "
              f"| Total: {total_pairs:,} | Time: {time.time()-batch_start:.2f}s")

        # Update result_dict
        for pid, user, cid, score in zip(
            interaction_df['profile_id'],
            interaction_df['username'],
            interaction_df['content_id'],
            preds
        ):
            result_dict.setdefault(pid, {'suggested_content': {}, 'user': {'username': user, 'profile_id': pid}})
            result_dict[pid]['suggested_content'][cid] = {
                'content_name': '', 'tag_names': '', 'type_id': '', 'score': float(score)
            }

    # Attach metadata
    content_clip_pl = pl.read_parquet(content_clip_path)
    content_unique = (
        content_clip_pl
        .unique(subset=['content_id'])
        .select(['content_id', 'content_name', 'tag_names', 'type_id'])
    )
    content_dict = {row[0]: (row[1], row[2], row[3]) for row in content_unique.iter_rows()}
    
    for pid, pdata in result_dict.items():
        for cid, cdata in pdata['suggested_content'].items():
            meta = content_dict.get(cid)
            if meta:
                cdata['content_name'], cdata['tag_names'], cdata['type_id'] = map(str, meta)

    # Ranking & rule assignment
    print("\nStarting parallel ranking & rule assignment...")
    rank_start = time.time()
    reordered_result = rank_result_parallel(result_dict, TOP_N, max_workers=os.cpu_count())
    result_with_rule = get_rulename_parallel(reordered_result, rule_info_path, tags_path)
    print(f"Ranking & rule assignment completed in {time.time()-rank_start:.2f}s")

    # Save outputs
    homepage_rule = []
    rule_content = ""
    for pid, info in result_with_rule.items():
        rulename_json_file = {'pid': pid, 'd': {}}
        rule_content += str(pid) + '|'
        for key, content in info['suggested_content'].items():
            if content['rule_id'] != -100:
                rulename_json_file['d'].setdefault(str(content['rule_id']), {})[key] = content['type_id']
        for rule, content_ids in rulename_json_file['d'].items():
            rule_content += str(rule) + ',' + ''.join(f'${v}#{k}' for k, v in content_ids.items()) + ';'
        rule_content = rule_content.rstrip(';') + '\n'
        rulename_json_file['d'] = ','.join(rulename_json_file['d'])
        homepage_rule.append(rulename_json_file)

    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_with_rule, f, indent=4, ensure_ascii=False)
    with open(rulename_json_path, 'w', encoding='utf-8') as f:
        json.dump(homepage_rule, f, indent=4, ensure_ascii=False)
    with open(rule_content_path, "w", encoding='utf-8') as f:
        f.write(rule_content)

    elapsed_time = time.time() - start_time
    print(f"\nElapsed time: {elapsed_time:.2f}s | Total pairs: {total_pairs:,} "
          f"| Avg per pair: {elapsed_time/total_pairs:.6f}s")
