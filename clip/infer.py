import os
import pandas as pd
import polars as pl
import torch
from tqdm import tqdm
import json
import time
from pathlib import Path

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
        for batch in tqdm(data, desc="Inference", leave=False):
            inputs = batch[0].to(device)
            outputs = model(inputs)
            predictions.extend(outputs['y_pred'].detach().cpu().numpy())
            torch.cuda.empty_cache()
    return predictions


if __name__ == "__main__":
    start_time = time.time()
    TOP_N = 200
    result_dict = {}
    homepage_rule = []
    rule_content = ""
    total_pairs = 0

    project_root = Path().resolve()
    os.makedirs(project_root / "clip" / "result", exist_ok=True)

    user_data_path = project_root / "month_mytv_info.parquet"
    clip_data_path = project_root / "mytv_vmp_content"
    content_clip_path = project_root / "clip/infer_data/merged_content_clips.parquet"
    tags_path = project_root / "tags"
    rule_info_path = project_root / "rule_info.parquet"
    result_dir = project_root / "clip/result"

    print("Loading model...")
    checkpoint_path = "model/clip/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    input_dim = checkpoint["model_state_dict"]["ECN.dfc.weight"].shape[1]
    model = DCNv3(input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Loading content clip metadata...")
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

    print("Running chunked inference...")
    chunks = process_infer_data(
        user_data_path, clip_data_path,
        num_user=80, num_clip=-1,
        user_batch_size=10
    )

    for idx, df in enumerate(chunks, 1):
        print(f"\n[Chunk {idx}] Rows: {len(df)}")
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

        partial_result = {}
        for pid, user, cid, score in zip(
            interaction_df['profile_id'],
            interaction_df['username'],
            interaction_df['content_id'],
            predictions
        ):
            partial_result.setdefault(pid, {})
            partial_result[pid].setdefault('suggested_content', {})
            partial_result[pid]['suggested_content'][cid] = {
                'content_name': '',
                'tag_names': '',
                'type_id': '',
                'score': float(score)
            }
            partial_result[pid]['user'] = {'username': user, 'profile_id': pid}

        for pid, pdata in partial_result.items():
            for cid, cdata in pdata['suggested_content'].items():
                meta = content_dict.get(cid)
                if meta:
                    cdata['content_name'], cdata['tag_names'], cdata['type_id'] = map(str, meta)

        ranked = rank_result(partial_result, TOP_N)
        ruled = get_rulename(ranked, rule_info_path, tags_path)

        for pid, info in ruled.items():
            result_dict[pid] = info

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

    print("\nSaving final outputs...")
    with open(result_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    with open(result_dir / "rulename.json", "w", encoding="utf-8") as f:
        json.dump(homepage_rule, f, indent=4, ensure_ascii=False)

    with open(result_dir / "rule_content.txt", "w", encoding="utf-8") as f:
        f.write(rule_content)

    elapsed = time.time() - start_time
    print(f"\n✅ Done. Total time: {elapsed:.2f}s | Total pairs: {total_pairs} | Avg/pair: {elapsed / total_pairs:.6f}s")