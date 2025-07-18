import os
import pandas as pd
import torch
from tqdm import tqdm
import json
import time
from pathlib import Path
from glob import glob

from item_process import process_clip_item
from user_process import process_user_data
from dcnv3 import DCNv3
from rule_process import get_rulename
from torch.utils.data import DataLoader, TensorDataset

def rank_result(data, n):
    reordered_data = {}
    for user_id in data:
        user_scores = [
            {
                'content_id': cid,
                'content_name': info['content_name'],
                'tag_names': info['tag_names'],
                'type_id': info['type_id'],
                'score': info['score']
            }
            for cid, info in data[user_id]['suggested_content'].items()
        ]
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

    user_data_path = project_root / "month_mytv_info.parquet"
    clip_data_path = project_root / "mytv_vmp_content"
    content_clip_path = project_root / "clip/infer_data/merged_content_clips.parquet"
    tags_path = project_root / "tags"
    rule_info_path = project_root / "rule_info.parquet"

    result_json_path = project_root / "clip/result/result.json"
    rulename_json_path = project_root / "clip/result/rulename.json"
    rule_content_path = project_root / "clip/result/rule_content.txt"

    user_df = process_user_data(user_data_path, "clip/infer_data", -1, mode='infer')
    clip_df = process_clip_item(clip_data_path, "clip/infer_data", -1, mode='infer')
    clip_df['content_id'] = clip_df['content_id'].astype(str)

    duration_files = glob(str(project_root / "clip/merged_duration/*.parquet"))
    user_profile_list = [
        pd.read_parquet(path)[['username', 'profile_id']].drop_duplicates()
        for path in duration_files
    ]
    user_profile_df = pd.concat(user_profile_list, ignore_index=True).drop_duplicates()
    user_profile_df = user_profile_df.merge(user_df, on='username', how='inner')

    checkpoint_path = "model/clip/best_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    expected_input_dim = checkpoint['model_state_dict']['ECN.dfc.weight'].shape[1]
    model = DCNv3(expected_input_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    result_dict = {}
    total_pairs = 0

    user_batch_size = 10
    infer_batch_size= 64
    print(f"User batch size: {user_batch_size}")
    print(f"Inference batch size: {infer_batch_size}")

    estimated_batches = (len(user_profile_df) + user_batch_size - 1) // user_batch_size
    print(f"Estimated number of user batches: {estimated_batches}")


    for i in range(0, len(user_profile_df), user_batch_size):
        batch_index = i // user_batch_size + 1

        print(f"\n📦 Processing batch {batch_index}/{estimated_batches} (Users {i} to {i + user_batch_size})")

        merge_start = time.time()

        user_batch = user_profile_df.iloc[i:i+user_batch_size].copy()
        user_batch = user_profile_df.iloc[i:i+user_batch_size].copy()
        user_batch['key'] = 1
        clip_df['key'] = 1
        cross_df = user_batch.merge(clip_df, on='key').drop('key', axis=1)

        print(f"Cross-merged size: {len(cross_df)} rows (≈ {len(user_batch)} users × {len(clip_df)} clips)")
        print(f"Merge time: {time.time() - merge_start:.2f} sec")

        interaction_df = cross_df[['username', 'content_id', 'profile_id']]
        exclude = {'username', 'content_id', 'profile_id'}
        features = cross_df.drop(columns=exclude)

        infer_tensor = torch.tensor(features.to_numpy(), dtype=torch.float32)
        infer_loader = DataLoader(TensorDataset(infer_tensor), batch_size=infer_batch_size, shuffle=False)

        infer_start = time.time()
        predictions = infer(model, infer_loader, device)
        infer_time = time.time() - infer_start
        print(f"Inference took {infer_time:.2f}s | Rows: {len(predictions)}")

        total_pairs += len(predictions)

        for pid, user, cid, score in zip(interaction_df['profile_id'],
                                         interaction_df['username'],
                                         interaction_df['content_id'],
                                         predictions):
            result_dict.setdefault(pid, {'suggested_content': {}, 'user': {'username': user, 'profile_id': pid}})
            result_dict[pid]['suggested_content'][cid] = {
                'content_name': '', 'tag_names': '', 'type_id': '', 'score': float(score)
            }

    content_clip_df = pd.read_parquet(content_clip_path)
    content_unique = content_clip_df.drop_duplicates(subset='content_id').set_index('content_id')

    for pid in result_dict:
        for cid in result_dict[pid]['suggested_content']:
            try:
                row = content_unique.loc[cid]
                result_dict[pid]['suggested_content'][cid]['content_name'] = row['content_name']
                result_dict[pid]['suggested_content'][cid]['tag_names'] = str(row['tag_names'])
                result_dict[pid]['suggested_content'][cid]['type_id'] = str(row['type_id'])
            except KeyError:
                continue

    reordered_result = rank_result(result_dict, TOP_N)
    result_with_rule = get_rulename(reordered_result, rule_info_path, tags_path)

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

    total_time = time.time() - start_time
    print("\nInference completed.")
    print(f"Total elapsed time: {total_time:.2f} seconds")
    print(f"Total user-item pairs inferred: {total_pairs}")
    print(f"Average time per pair: {total_time / total_pairs:.6f} seconds")