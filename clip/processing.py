import os
import time
import pandas as pd
import polars as pl
import torch
from tqdm import tqdm
import json
from pathlib import Path
import glob
from math import ceil
import logging

from torch.utils.data import DataLoader, TensorDataset
from dcnv3 import DCNv3
from rule_process import get_rulename
from user_process import process_user_data
from item_process import process_movie_item

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

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

def save_chunk_results(temp_dict, temp_file):
    with open(temp_file, 'a', encoding='utf-8') as f:
        for pid, data in temp_dict.items():
            json.dump({pid: data}, f, ensure_ascii=False)
            f.write('\n')

if __name__ == "__main__":
    start_time = time.time()
    TOP_N = 200

    project_root = Path("/kaggle/working/MyRecSys/clip").resolve()
    os.makedirs(project_root / "movie" / "result", exist_ok=True)
    os.makedirs(project_root / "movie" / "infer_data", exist_ok=True)

    # Define file paths
    user_data_path = os.path.join(project_root, "month_mytv_info.parquet")
    clip_data_path = os.path.join(project_root, "mytv_vmp_content")
    processed_user_path = os.path.join(project_root, "movie/infer_data/user_data.parquet")
    processed_item_path = os.path.join(project_root, "movie/infer_data/movie_item_data.parquet")
    content_movie_path = os.path.join(project_root, "movie/infer_data/merged_content_movies.parquet")
    tags_path = os.path.join(project_root, "tags")
    rule_info_path = os.path.join(project_root, "rule_info.parquet")
    result_json_path = os.path.join(project_root, "movie/result/result.json")
    rulename_json_path = os.path.join(project_root, "movie/result/rulename.json")
    rule_content_path = os.path.join(project_root, "movie/result/rule_content.txt")
    temp_results_path = os.path.join(project_root, "movie/infer_data/temp_results.jsonl")

    # Preprocess user and item data
    logger.info("=== Preprocessing User and Item Data ===")
    preprocess_start = time.time()
    try:
        if not os.path.exists(processed_user_path):
            logger.info("  Processing user data...")
            user_process_start = time.time()
            process_user_data(user_data_path, output_dir="movie/infer_data", num_user=-1, mode='infer')
            user_process_time = time.time() - user_process_start
            logger.info(f"  User data processed in {user_process_time:.2f} seconds")
        else:
            logger.info("  User data already exists, skipping preprocessing.")

        if not os.path.exists(processed_item_path):
            logger.info("  Processing item data...")
            item_process_start = time.time()
            process_movie_item(movie_data_path, output_dir="movie/infer_data", num_movie=-1, mode='infer')
            item_process_time = time.time() - item_process_start
            logger.info(f"  Item data processed in {item_process_time:.2f} seconds")
        else:
            logger.info("  Item data already exists, skipping preprocessing.")
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Ensure raw data files (month_mytv_info.parquet, mytv_vmp_content) and encoders/scalers are available.")
        exit(1)
    preprocess_time = time.time() - preprocess_start
    logger.info(f"Preprocessing completed in {preprocess_time:.2f} seconds\n")

    # Load preprocessed data
    logger.info("=== Loading User and Item Data ===")
    load_start = time.time()
    try:
        user_df = pd.read_parquet(processed_user_path)
        clip_df = pd.read_parquet(processed_item_path)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Preprocessing failed to generate required parquet files.")
        exit(1)

    duration_dir = os.path.join(project_root, "movie/merged_duration")
    if not os.path.exists(duration_dir):
        logger.error(f"Error: Duration directory not found at {duration_dir}")
        logger.error("Run merge_parquet_files from duration_process.py to generate duration files.")
        exit(1)
    durations = glob.glob(os.path.join(duration_dir, "*.parquet"))
    if not durations:
        logger.error(f"Error: No duration data found in {duration_dir}")
        logger.error("Run merge_parquet_files from duration_process.py to generate duration files.")
        exit(1)
    user_profile_list = []
    for duration in durations:
        df = pd.read_parquet(duration, columns=["username", "profile_id"])
        user_profile_list.append(df.drop_duplicates())
    user_profile_df = pd.concat(user_profile_list, ignore_index=True).drop_duplicates()
    user_profile_df = user_profile_df.merge(user_df, on="username", how="inner")
    load_time = time.time() - load_start
    logger.info(f"Data loaded in {load_time:.2f} seconds")
    logger.info(f"Loaded {len(user_profile_df)} users and {len(movie_df)} items\n")

    # Load model
    logger.info("=== Loading Model ===")
    model_load_start = time.time()
    checkpoint_path = os.path.join(project_root, "model/movie/best_model.pth")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        logger.error(f"Error: Model checkpoint not found at {checkpoint_path}")
        exit(1)
    expected_input_dim = checkpoint["model_state_dict"]["ECN.dfc.weight"].shape[1]
    model = DCNv3(expected_input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model_load_time = time.time() - model_load_start
    logger.info(f"Model loaded in {model_load_time:.2f} seconds\n")

    # Process users in chunks
    user_batch_size = 50
    num_users = len(user_profile_df)
    num_chunks = ceil(num_users / user_batch_size)
    total_pairs = 0
    chunk_times = []

    logger.info(f"=== Processing {num_users} Users in {num_chunks} Chunks ===")
    # Clear temporary results file if it exists
    if os.path.exists(temp_results_path):
        os.remove(temp_results_path)

    for i in range(num_chunks):
        chunk_start = time.time()
        logger.info(f"[Chunk {i+1}/{num_chunks}]")

        # Select user chunk
        select_start = time.time()
        start_idx = i * user_batch_size
        end_idx = min((i+1) * user_batch_size, num_users)
        user_chunk = user_profile_df.iloc[start_idx:end_idx]
        select_time = time.time() - select_start
        logger.info(f"  Select users: {select_time:.2f} seconds ({len(user_chunk)} users)")

        # Perform cross join
        cross_start = time.time()
        user_chunk_pl = pl.from_pandas(user_chunk)
        clip_pl = pl.from_pandas(movie_df)
        cross_df = user_chunk_pl.join(movie_pl, how="cross")
        cross_df_pd = cross_df.to_pandas()
        cross_time = time.time() - cross_start
        logger.info(f"  Cross join: {cross_time:.2f} seconds ({len(cross_df)} pairs)")

        # Prepare data for inference
        prep_start = time.time()
        exclude = {'username', 'content_id', 'profile_id'}
        to_convert = [col for col in cross_df_pd.columns if col not in exclude]
        cross_df_pd[to_convert] = cross_df_pd[to_convert].apply(pd.to_numeric, errors='coerce')
        cross_df_pd = cross_df_pd.dropna()
        cross_df_pd = cross_df_pd.astype({col: 'float32' for col in to_convert})
        interaction_df = cross_df_pd[['username', 'content_id', 'profile_id']]
        features = cross_df_pd.drop(columns=['username', 'content_id', 'profile_id'])
        prep_time = time.time() - prep_start
        logger.info(f"  Data preparation: {prep_time:.2f} seconds")

        # Create tensor
        tensor_start = time.time()
        infer_tensor = torch.tensor(features.to_numpy(), dtype=torch.float32)
        infer_loader = DataLoader(TensorDataset(infer_tensor), batch_size=2048, shuffle=False)
        tensor_time = time.time() - tensor_start
        logger.info(f"  Tensor creation: {tensor_time:.2f} seconds")

        # Run inference
        infer_start = time.time()
        predictions = infer(model, infer_loader, device)
        total_pairs += len(predictions)
        infer_time = time.time() - infer_start
        logger.info(f"  Inference: {infer_time:.2f} seconds")

        # Collect top N results per user
        collect_start = time.time()
        temp_dict = {}
        for pid, user, cid, score in zip(
            interaction_df['profile_id'],
            interaction_df['username'],
            interaction_df['content_id'],
            predictions
        ):
            if pid not in temp_dict:
                temp_dict[pid] = []
            temp_dict[pid].append((cid, float(score)))

        result_dict = {}
        for pid in temp_dict:
            sorted_items = sorted(temp_dict[pid], key=lambda x: x[1], reverse=True)[:TOP_N]
            result_dict[pid] = {
                'suggested_content': {
                    cid: {
                        'content_name': '',
                        'tag_names': '',
                        'type_id': '',
                        'score': score
                    } for cid, score in sorted_items
                },
                'user': {
                    'username': user_profile_df.loc[user_profile_df['profile_id'] == pid, 'username'].iloc[0],
                    'profile_id': pid
                }
            }
        collect_time = time.time() - collect_start
        logger.info(f"  Collect top {TOP_N}: {collect_time:.2f} seconds")

        # Save chunk results to disk
        save_start = time.time()
        save_chunk_results(result_dict, temp_results_path)
        save_time = time.time() - save_start
        logger.info(f"  Save chunk results: {save_time:.2f} seconds")

        chunk_time = time.time() - chunk_start
        chunk_times.append(chunk_time)
        logger.info(f"  Total chunk time: {chunk_time:.2f} seconds | Pairs: {len(predictions):,}\n")

    # Add content metadata
    logger.info("=== Adding Content Metadata ===")
    meta_start = time.time()
    try:
        content_movie_pl = pl.read_parquet(content_movie_path)
    except FileNotFoundError:
        logger.error(f"Error: Content metadata not found at {content_movie_path}")
        exit(1)
    content_unique = (
        content_movie_pl
        .unique(subset=['content_id'])
        .select(['content_id', 'content_name', 'tag_names', 'type_id'])
    )
    content_dict = {row[0]: (row[1], row[2], row[3]) for row in content_unique.iter_rows()}

    result_dict = {}
    with open(temp_results_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            for pid, pdata in data.items():
                result_dict[pid] = pdata
                for cid, cdata in pdata['suggested_content'].items():
                    meta = content_dict.get(cid)
                    if meta:
                        cdata['content_name'], cdata['tag_names'], cdata['type_id'] = map(str, meta)
    meta_time = time.time() - meta_start
    logger.info(f"Content metadata added in {meta_time:.2f} seconds\n")

    # Ranking and rule assignment
    logger.info("=== Ranking and Rule Assignment ===")
    rank_start = time.time()
    reordered_result = rank_result(result_dict, TOP_N)
    try:
        result_with_rule = get_rulename(reordered_result, rule_info_path, tags_path)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Ensure rule_info.parquet and tags/ directory are available.")
        exit(1)
    rank_time = time.time() - rank_start
    logger.info(f"Ranking and rule assignment completed in {rank_time:.2f} seconds\n")

    # Save results
    logger.info("=== Saving Results ===")
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
    save_time = time.time() - save_start
    logger.info(f"Results saved in {save_time:.2f} seconds\n")

    # Final timing summary
    total_time = time.time() - start_time
    logger.info("=== Final Timing Summary ===")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Total user-item pairs: {total_pairs:,}")
    logger.info(f"Average inference time per pair: {total_time / total_pairs:.6f} seconds")
    logger.info(f"Average chunk time: {sum(chunk_times) / len(chunk_times):.2f} seconds")
    logger.info(f"Number of chunks processed: {len(chunk_times)}")