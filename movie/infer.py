import os
import pandas as pd
import torch
from tqdm import tqdm
import json
import time
from pathlib import Path
from glob import glob

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

    project_root = Path().resolve()
    os.makedirs(project_root / "movie" / "result", exist_ok=True)
    os.makedirs(project_root / "movie" / "infer_data", exist_ok=True)

    user_data_path = os.path.join(project_root, "month_mytv_info.parquet")
    movie_data_path = os.path.join(project_root, "mytv_vmp_content")
    content_movie_path = os.path.join(project_root, "movie/infer_data/merged_content_movies.parquet")
    tags_path = os.path.join(project_root, "tags")
    rule_info_path = os.path.join(project_root, "rule_info.parquet")

    result_json_path = os.path.join(project_root, "movie/result/result.json")
    rulename_json_path = os.path.join(project_root, "movie/result/rulename.json")
    rule_content_path = os.path.join(project_root, "movie/result/rule_content.txt")

    part_files = sorted(glob(str(project_root / "movie/infer_data/infer_user_movie/infer_user_movie_part_*.parquet")))
    if not part_files:
        process_infer_data(user_data_path, movie_data_path, num_user=-1, num_movie=-1, output_dir_path="movie/infer_data",
                        user_batch_size=100, chunk_size=None, max_files=500)
        part_files = sorted(glob(str(project_root / "movie/infer_data/infer_user_movie/infer_user_movie_part_*.parquet")))

    checkpoint_path = "model/movie/best_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    expected_input_dim = checkpoint["model_state_dict"]["ECN.dfc.weight"].shape[1]
    model = DCNv3(expected_input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    result_dict = {}
    total_pairs = 0

    for part_file in part_files:
        df = pd.read_parquet(part_file).fillna(0)
        exclude = {'username', 'content_id', 'profile_id'}
        to_convert = [col for col in df.columns if col not in exclude]
        df[to_convert] = df[to_convert].apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        df = df.astype({col: 'float32' for col in to_convert})

        interaction_df = df[['username', 'content_id', 'profile_id']]
        features = df.drop(columns=['username', 'content_id', 'profile_id'])

        infer_tensor = torch.tensor(features.to_numpy(), dtype=torch.float32)
        infer_loader = DataLoader(TensorDataset(infer_tensor), batch_size=2048, shuffle=False)

        predictions = infer(model, infer_loader, device)
        total_pairs += len(predictions)

        for pid, user, cid, score in zip(interaction_df['profile_id'],
                                        interaction_df['username'],
                                        interaction_df['content_id'],
                                        predictions):
            result_dict.setdefault(pid, {})
            result_dict[pid].setdefault('suggested_content', {})
            result_dict[pid]['suggested_content'][cid] = {
                'content_name': '', 'tag_names': '', 'type_id': '', 'score': float(score)
            }
            result_dict[pid]['user'] = {'username': user, 'profile_id': pid}

    content_movie_df = pd.read_parquet(content_movie_path)
    content_unique = content_movie_df.drop_duplicates(subset='content_id').set_index('content_id')

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

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(f"Total user-item pairs: {total_pairs}")
    print(f"Average inference time per pair: {elapsed_time / total_pairs:.6f} seconds")
