import os
import pandas as pd
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
        
        # Collect all content scores for the user
        for content_id in data[user_id]['suggested_content']:
            content_name = data[user_id]['suggested_content'][content_id]['content_name']
            tag_names = data[user_id]['suggested_content'][content_id]['tag_names']
            type_id = data[user_id]['suggested_content'][content_id]['type_id']
            score = data[user_id]['suggested_content'][content_id]['score']
            
            user_scores.append({
                'content_id': content_id,
                'content_name': content_name,
                'tag_names': tag_names,
                'type_id': type_id,
                'score': score
            })

        # Sort the user scores in descending order by score (top-n means highest scores first)
        sorted_user_scores = sorted(user_scores, key=lambda x: x['score'], reverse=True)

        # Keep only the top-n items
        top_n_user_scores = sorted_user_scores[:n]

        # Rebuild the sorted suggested content with only the top-n items
        sorted_suggested_content = {}
        for film in top_n_user_scores:
            sorted_suggested_content[film['content_id']] = {
                'content_name': film['content_name'],
                'tag_names': film['tag_names'],
                'type_id': film['type_id'],
                'score': film['score']
            }

        # Store the sorted data for the user
        reordered_data[user_id] = {
            'suggested_content': sorted_suggested_content,
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

    return predictions


if __name__ == "__main__":
    start_time = time.time()

    TOP_N = 200

    user_data_path = "month_mytv_info.parquet"
    clip_data_path = "mytv_vmp_content"
    tags_path = "tags"
    rule_info_path = "rule_info.parquet"

    project_root = Path().resolve()

    os.makedirs(project_root / "clip" / "result", exist_ok=True)
    os.makedirs(project_root / "clip" / "infer_data", exist_ok=True)

    infer_user_clip_path = os.path.join(project_root, "clip/infer_data/infer_user_clip.csv")
    result_json_path = os.path.join(project_root, "clip/result/result.json")
    rulename_json_path = os.path.join(project_root, "clip/result/rulename.json")
    content_clip_path = os.path.join(project_root, "clip/infer_data/merged_content_clips.csv")
    rule_content_path = os.path.join(project_root, "clip/result/rule_content.txt")

    user_data_path = os.path.join(project_root, user_data_path)
    clip_data_path = os.path.join(project_root, clip_data_path)
    tags_path = os.path.join(project_root, tags_path)
    rule_info_path = os.path.join(project_root, rule_info_path)

    if os.path.exists(infer_user_clip_path):
        infer_user_clip_df = pd.read_csv(infer_user_clip_path, low_memory=False)
    else:
        process_infer_data(user_data_path, clip_data_path, 10, 10000, infer_user_clip_path)
        infer_user_clip_df = pd.read_csv(infer_user_clip_path, low_memory=False)

    infer_user_clip_df = infer_user_clip_df.fillna(0)
    exclude = {'username', 'content_id', 'profile_id'}
    to_convert = [col for col in infer_user_clip_df.columns if col not in exclude]
    infer_user_clip_df[to_convert] = infer_user_clip_df[to_convert].apply(pd.to_numeric, errors='coerce')
    infer_user_clip_df = infer_user_clip_df.dropna()
    infer_user_clip_df = infer_user_clip_df.astype({col: 'float32' for col in infer_user_clip_df.columns if col not in ['username', 'content_id', 'profile_id']})

    interaction_df = infer_user_clip_df[['username', 'content_id', 'profile_id']]
    infer_df = infer_user_clip_df.drop(columns = ['username', 'content_id', 'profile_id'], axis=1)

    infer_data = infer_df.to_numpy()
    infer_data_tensor = torch.tensor(infer_data, dtype=torch.float32)
    infer_dataset = TensorDataset(infer_data_tensor)
    infer_loader = DataLoader(infer_dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = "model/clip/best_model.pth"              
    checkpoint      = torch.load(checkpoint_path, map_location=device)
    expected_input_dim = checkpoint["model_state_dict"]["ECN.dfc.weight"].shape[1]
    model = DCNv3(expected_input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    result = infer(model, infer_loader, device)
    content_clip_df = pd.read_csv(content_clip_path)

    content_unique = (
        content_clip_df
        .drop_duplicates(subset='content_id', keep='first')
        .set_index('content_id')
    )

    result_dict = {}
    for pid, user, cid, score in zip(
        interaction_df['profile_id'],
        interaction_df['username'],
        interaction_df['content_id'],
        result
    ):
        result_dict.setdefault(pid, {})

        cid_key = cid
        result_dict[pid].setdefault('suggested_content', {})
        result_dict[pid]['suggested_content'][cid_key] = {
            'content_name': content_unique.at[cid, 'content_name'],
            'tag_names': str(content_unique.at[cid, 'tag_names']),
            'type_id': str(content_unique.at[cid, 'type_id']),
            'score': float(score)
        }

        result_dict[pid]['user'] = {
            'username': user,
            'profile_id': pid
        }

    reordered_result = rank_result(result_dict, TOP_N)
    result_with_rule = get_rulename(reordered_result, rule_info_path, tags_path)
    
    homepage_rule = []
    rule_content = ""

    for pid in result_with_rule:
        rulename_json_file = {}
        rulename_json_file['pid'] = pid
        rulename_json_file['d'] = dict()

        rule_content += str(pid) + '|'
        content_dict = {}
        for key,content in result_with_rule[pid]['suggested_content'].items():
            if content['rule_id'] != -100:
                if not rulename_json_file['d'].get(str(content['rule_id'])):
                    rulename_json_file['d'][str(content['rule_id'])] = {}
                rulename_json_file['d'][str(content['rule_id'])][key] = content['type_id']

        for rule in rulename_json_file['d']:
            rule_content += str(rule) + ','
            for content_id,type_id in rulename_json_file['d'][rule].items():
                rule_content += f'${type_id}#{content_id}'
            rule_content += ';'
        rule_content = rule_content.rstrip(';')
        rule_content += '\n'
        rulename_json_file['d'] = ','.join(rulename_json_file['d'])  
        homepage_rule.append(rulename_json_file)
    

    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_with_rule, f, indent=4, ensure_ascii=False)
    
    with open(rulename_json_path, 'w', encoding='utf-8') as f:
        json.dump(homepage_rule, f, indent=4, ensure_ascii=False)
    with open(rule_content_path, "w", encoding='utf-8') as f:
        f.write(rule_content)

    end_time = time.time()
    elapsed_time = end_time - start_time
    total_pairs = len(interaction_df)
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(f"Total user-item pairs: {total_pairs}")
    print(f"Average inference time per pair: {elapsed_time / total_pairs:.6f} seconds")
