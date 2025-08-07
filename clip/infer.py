import os
import pandas as pd
import polars as pl
import torch
from tqdm import tqdm
import json
import time
from pathlib import Path
from glob import glob

from concurrent.futures import ProcessPoolExecutor
from processing import process_infer_data
from torch.utils.data import DataLoader, TensorDataset
from dcnv3 import DCNv3
from rule_process import get_rulename

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

    # Setup paths
    project_root = Path().resolve()
    os.makedirs(project_root / "clip" / "result", exist_ok=True)
    os.makedirs(project_root / "clip" / "infer_data", exist_ok=True)

    user_data_path = project_root / "month_mytv_info.parquet"
    clip_data_path = project_root / "mytv_vmp_content"
    content_clip_path = project_root / "clip/infer_data/merged_content_clips.parquet"
    tags_path = project_root / "tags"
    rule_info_path = project_root / "rule_info.parquet"

    result_json_path = project_root / "clip/result/result.json"
    rulename_json_path = project_root / "clip/result/rulename.json"
    rule_content_path = project_root / "clip/result/rule_content.txt"

    # Load model
    print("Loading model...")
    checkpoint_path = "model/clip/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    input_dim = checkpoint["model_state_dict"]["ECN.dfc.weight"].shape[1]

    model = DCNv3(input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    result_dict = {}
    total_pairs = 0

    # Inference preparation
    print("Generating user-clip cross joins in memory...")
    chunks = process_infer_data(
        user_data_path, clip_data_path,
        num_user=80, num_clip=-1,
        output_dir_path="clip/infer_data",
        user_batch_size=10,
        chunk_size=None,
        max_files=-1
    )

    # Inference loop
    for idx, df in enumerate(chunks, 1):
        print(f"[Inference {idx}/{len(chunks)}] Processing {len(df)} rows")

        exclude = {'username', 'content_id', 'profile_id'}
        to_convert = [col for col in df.columns if col not in exclude]
        df[to_convert] = df[to_convert].apply(pd.to_numeric, errors='coerce')
        df = df.dropna().astype({col: 'float32' for col in to_convert})

        interaction_df = df[['username', 'content_id', 'profile_id']]
        features = df.drop(columns=['username', 'content_id', 'profile_id'])

        infer_tensor = torch.tensor(features.to_numpy(), dtype=torch.float32)
        infer_loader = DataLoader(TensorDataset(infer_tensor), batch_size=2048, shuffle=False)

        predictions = infer(model, infer_loader, device)
        total_pairs += len(predictions)

        for pid, user, cid, score in zip(
            interaction_df['profile_id'],
            interaction_df['username'],
            interaction_df['content_id'],
            predictions
        ):
            result_dict.setdefault(pid, {})
            result_dict[pid].setdefault('suggested_content', {})
            result_dict[pid]['suggested_content'][cid] = {
                'content_name': '',
                'tag_names': '',
                'type_id': '',
                'score': float(score)
            }
            result_dict[pid]['user'] = {'username': user, 'profile_id': pid}

    # Postprocessing: enrich content info
    print("\nStarting ranking and rule assignment...")
    rank_start = time.time()

    content_clip_pl = pl.read_parquet(content_clip_path)
    content_unique = (
        content_clip_pl
        .unique(subset=['content_id'])
        .select(['content_id', 'content_name', 'tag_names', 'type_id'])
    )
    content_dict = {
        row[0]: (row[1], row[2], row[3])
        for row in content_unique.iter_rows()
    }

    for pid, pdata in result_dict.items():
        for cid, cdata in pdata['suggested_content'].items():
            meta = content_dict.get(cid)
            if meta:
                cdata['content_name'], cdata['tag_names'], cdata['type_id'] = map(str, meta)

    print("\nRanking top results...")
    reordered_result = rank_result(result_dict, TOP_N)
    print(f"Ranking completed in {time.time()-rank_start:.2f}s")

    print("\nAssigning rules...")
    rule_start = time.time()
    result_with_rule = get_rulename(reordered_result, rule_info_path, tags_path)
    print(f"Rule assignment completed in {time.time()-rule_start:.2f}s")

    # Prepare output files
    homepage_rule = []
    rule_content = ""

    for pid, info in result_with_rule.items():
        rulename_json_file = {'pid': pid, 'd': {}}
        rule_content += str(pid) + '|'

        for cid, content in info['suggested_content'].items():
            if content['rule_id'] != -100:
                rulename_json_file['d'].setdefault(str(content['rule_id']), {})[cid] = content['type_id']

        for rule, content_ids in rulename_json_file['d'].items():
            rule_content += str(rule) + ',' + ''.join(f'${v}#{k}' for k, v in content_ids.items()) + ';'

        rule_content = rule_content.rstrip(';') + '\n'
        rulename_json_file['d'] = ','.join(rulename_json_file['d'])
        homepage_rule.append(rulename_json_file)

    # Write output files
    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_with_rule, f, indent=4, ensure_ascii=False)
    with open(rulename_json_path, 'w', encoding='utf-8') as f:
        json.dump(homepage_rule, f, indent=4, ensure_ascii=False)
    with open(rule_content_path, 'w', encoding='utf-8') as f:
        f.write(rule_content)

    # Final logs
    elapsed_time = time.time() - start_time
    print(f"\nElapsed time: {elapsed_time:.2f}s")
    print(f"Total user-item pairs: {total_pairs}")
    print(f"Average inference time per pair: {elapsed_time / total_pairs:.6f} seconds")
