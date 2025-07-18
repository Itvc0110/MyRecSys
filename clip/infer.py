import os
import polars as pl
import torch
from tqdm import tqdm
import json
import time
from pathlib import Path
from glob import glob
import pandas as pd

from item_process import process_clip_item, transform_item_data
from user_process import process_user_data, transform_user_data
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

    user_df_raw = process_user_data(user_data_path, "clip/infer_data", -1, mode='infer')
    clip_df_raw = process_clip_item(clip_data_path, "clip/infer_data", -1, mode='infer')

    # Apply saved encoders (same as used in training)
    user_df_encoded = transform_user_data(user_df_raw, ["province", "package_code"])
    clip_df_encoded = transform_item_data(clip_df_raw,
        ["content_country", "locked_level", "VOD_CODE", "contract", "type_id"],
        "content_cate_id"
    )

    user_df = pl.from_pandas(user_df_encoded)
    clip_df = pl.from_pandas(clip_df_encoded)

    clip_df = clip_df.with_columns(pl.col("content_id").cast(pl.Utf8))

    duration_files = glob(str(project_root / "clip/merged_duration/*.parquet"))
    user_profile_list = [
        pl.read_parquet(path).select(["username", "profile_id"]).unique()
        for path in duration_files
    ]
    user_profile_df = pl.concat(user_profile_list).unique()
    uuser_profile_df = user_profile_df.join(user_df, on="username", how="inner")

    checkpoint_path = "model/clip/best_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    expected_input_dim = checkpoint['model_state_dict']['ECN.dfc.weight'].shape[1]
    model = DCNv3(expected_input_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    result_dict = {}
    total_pairs = 0

    user_batch_size = 1
    infer_batch_size= 64
    print(f"User batch size: {user_batch_size}")
    print(f"Inference batch size: {infer_batch_size}")

    estimated_batches = (len(user_profile_df) + user_batch_size - 1) // user_batch_size
    print(f"Estimated number of user batches: {estimated_batches}")


    for i in range(0, len(user_profile_df), user_batch_size):
        batch_index = i // user_batch_size + 1

        print(f"\nProcessing batch {batch_index}/{estimated_batches} (Users {i} to {i + user_batch_size})")

        merge_start = time.time()

        user_batch = user_profile_df.slice(i, user_batch_size)
        user_batch = user_batch.with_columns(pl.lit(1).alias("key"))
        clip_df_batch = clip_df.with_columns(pl.lit(1).alias("key"))
        cross_df = user_batch.join(clip_df_batch, on="key").drop("key")

        print(f"Cross-merged size: {len(cross_df)} rows (≈ {len(user_batch)} users × {len(clip_df)} clips)")
        print(f"Merge time: {time.time() - merge_start:.2f} sec")

        interaction_df = cross_df.select(['username', 'content_id', 'profile_id']).to_pandas()
        feature_df = cross_df.drop(['username', 'content_id', 'profile_id'])

        for col in feature_df.columns:
            feature_df = feature_df.with_columns(
                pl.col(col).cast(pl.Float32, strict=False)
            )

        feature_df = feature_df.drop_nulls()

        features_np = feature_df.to_numpy()
        ##########################################
        assert features_np.shape[1] == expected_input_dim, \
        f"Expected input dim: {expected_input_dim}, but got: {features_np.shape[1]}"
        #########################################
        infer_tensor = torch.tensor(features_np, dtype=torch.float32)
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

    content_clip_df = pl.read_parquet(content_clip_path)
    content_unique = content_clip_df.unique(subset='content_id').to_pandas().set_index("content_id")

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