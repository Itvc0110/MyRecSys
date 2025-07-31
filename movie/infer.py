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
from rule_process import get_rulename_parallel

def _rank_user(args):
    user_id, suggested_content, top_n = args
    user_scores = [
        (cid, content['score'], content)
        for cid, content in suggested_content.items()
    ]
    top_items = sorted(user_scores, key=lambda x: x[1], reverse=True)[:top_n]
    return user_id, {
        'suggested_content': {cid: content for cid, _, content in top_items}
    }

def rank_result_parallel(data, n, max_workers=None):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        args = [(uid, user_data['suggested_content'], n) for uid, user_data in data.items()]
        results = list(executor.map(_rank_user, args))
    return {uid: {'suggested_content': sc, 'user': data[uid]['user']} for uid, sc in results}

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
        process_infer_data(user_data_path, movie_data_path, num_user=40, num_movie=-1, output_dir_path="movie/infer_data",
                        user_batch_size=40, chunk_size=None, max_files=-1)
        part_files = sorted(glob(str(project_root / "movie/infer_data/infer_user_movie/infer_user_movie_part_*.parquet")))

    checkpoint_path = "model/movie/best_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    expected_input_dim = checkpoint["model_state_dict"]["ECN.dfc.weight"].shape[1]
    model = DCNv3(expected_input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    result_dict = {}
    total_pairs = 0

    for idx, part_file in enumerate(part_files, 1):
        file_start = time.time()
        print(f"[Inference {idx}/{len(part_files)}] Loading {os.path.basename(part_file)}...")

        df = pl.read_parquet(part_file).fill_null(0).to_pandas()
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

        elapsed_file = time.time() - file_start
        print(f"  ↪︎ Finished {os.path.basename(part_file)} "
              f"| {len(df):,} rows | {elapsed_file:.2f}s "
              f"| Cumulative pairs: {total_pairs:,}")

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


    print("\nStarting ranking and rule assignment...")
    rank_start = time.time()

    content_movie_pl = pl.read_parquet(content_movie_path)
    content_unique = (
        content_movie_pl
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

    print("\nStarting parallel ranking...")
    rank_start = time.time()
    reordered_result = rank_result_parallel(result_dict, TOP_N, max_workers=os.cpu_count())
    print(f"Ranking completed in {time.time()-rank_start:.2f}s")

    print("\nStarting parallel rule assignment...")
    rule_start = time.time()
    result_with_rule = get_rulename_parallel(reordered_result, rule_info_path, tags_path)
    print(f"Rule assignment completed in {time.time()-rule_start:.2f}s")

    print(f"Ranking & rule assignment completed in {time.time()-rank_start:.2f}s")

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
