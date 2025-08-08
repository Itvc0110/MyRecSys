import os
import pandas as pd
import polars as pl
import torch
from tqdm import tqdm
import json
from pathlib import Path
import glob
from math import ceil
import logging
import time
import sys

from torch.utils.data import DataLoader, TensorDataset
from dcnv3 import DCNv3
from rule_process import get_rulename
from user_process import process_user_data
from item_process import process_clip_item, process_clip_item

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(sys.stdout)])
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

def save_chunk_results(temp_dict, result_file, rule_content_file, homepage_rule, content_dict, rule_info_path, tags_path, top_n):
    logger.info("  Saving chunk results...")
    chunk_result = {}
    for pid, data in temp_dict.items():
        for cid, cdata in data['suggested_content'].items():
            meta = content_dict.get(cid)
            if meta:
                cdata['content_name'], cdata['tag_names'], cdata['type_id'] = map(str, meta)
        chunk_result[pid] = data

    reordered_chunk = rank_result(chunk_result, top_n)
    result_with_rule = get_rulename(reordered_chunk, rule_info_path, tags_path)

    with open(result_file, 'a', encoding='utf-8') as f:
        for pid, data in result_with_rule.items():
            json.dump({pid: data}, f, ensure_ascii=False)
            f.write('\n')

    with open(rule_content_file, 'a', encoding='utf-8') as f:
        for pid, info in result_with_rule.items():
            rulename_json_file = {'pid': pid, 'd': {}}
            rule_content = f"{pid}|"
            for key, content in info['suggested_content'].items():
                if content['rule_id'] != -100:
                    rulename_json_file['d'].setdefault(str(content['rule_id']), {})[key] = content['type_id']
            for rule, content_ids in rulename_json_file['d'].items():
                rule_content += f"{rule}," + ''.join(f'${v}#{k}' for k, v in content_ids.items()) + ';'
            rule_content = rule_content.rstrip(';') + '\n'
            f.write(rule_content)
            rulename_json_file['d'] = ','.join(rulename_json_file['d'])
            homepage_rule.append(rulename_json_file)

if __name__ == "__main__":
    print("Starting script...")
    start_time = time.time()
    TOP_N = 200
    CONTENT_TYPE = "clip"  # Set to "clip" for clip data

    project_root = Path("/kaggle/working/MyRecSys/clip").resolve()
    content_dir = CONTENT_TYPE
    os.makedirs(project_root / content_dir / "result", exist_ok=True)
    os.makedirs(project_root / content_dir / "infer_data", exist_ok=True)

    # Define file paths
    user_data_path = os.path.join(project_root, "month_mytv_info.parquet")
    item_data_path = os.path.join(project_root, "mytv_vmp_content")
    processed_user_path = os.path.join(project_root, f"{content_dir}/infer_data/user_data.parquet")
    processed_item_path = os.path.join(project_root, f"{content_dir}/infer_data/{content_dir}_item_data.parquet")
    content_path = os.path.join(project_root, f"{content_dir}/infer_data/merged_content_{content_dir}s.parquet")
    tags_path = os.path.join(project_root, "tags")
    rule_info_path = os.path.join(project_root, "rule_info.parquet")
    result_json_path = os.path.join(project_root, f"{content_dir}/result/result.json")
    result_jsonl_path = os.path.join(project_root, f"{content_dir}/result/result.jsonl")
    rulename_json_path = os.path.join(project_root, f"{content_dir}/result/rulename.json")
    rule_content_path = os.path.join(project_root, f"{content_dir}/result/rule_content.txt")
    duration_dir = os.path.join(project_root, f"{content_dir}/merged_duration")
    model_path = os.path.join(project_root, f"model/{content_dir}/best_model.pth")

    # Preprocess user and item data
    logger.info("=== Preprocessing User and Item Data ===")
    preprocess_start = time.time()
    try:
        if not os.path.exists(processed_user_path):
            logger.info("  Processing user data...")
            user_process_start = time.time()
            process_user_data(user_data_path, output_dir=f"{content_dir}/infer_data", num_user=-1, mode='infer')
            user_process_time = time.time() - user_process_start
            logger.info(f"  User data processed in {user_process_time:.2f} seconds")
        else:
            logger.info("  User data already exists, skipping preprocessing.")

        if not os.path.exists(processed_item_path):
            logger.info("  Processing item data...")
            item_process_start = time.time()
            process_item = process_clip_item if CONTENT_TYPE == "clip" else process_clip_item
            process_item(item_data_path, output_dir=f"{content_dir}/infer_data", num_clip=-1 if CONTENT_TYPE == "clip" else -1, mode='infer')
            item_process_time = time.time() - item_process_start
            logger.info(f"  Item data processed in {item_process_time:.2f} seconds")
        else:
            logger.info("  Item data already exists, skipping preprocessing.")
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        exit(1)
    preprocess_time = time.time() - preprocess_start
    logger.info(f"Preprocessing completed in {preprocess_time:.2f} seconds\n")

    # Load preprocessed data
    logger.info("=== Loading User and Item Data ===")
    load_start = time.time()
    try:
        user_df = pd.read_parquet(processed_user_path)
        item_df = pd.read_parquet(processed_item_path)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        exit(1)

    if len(user_df) == 0:
        logger.error("Error: user_df is empty")
        exit(1)
    if len(item_df) == 0:
        logger.error("Error: item_df is empty")
        exit(1)

    if not os.path.exists(duration_dir):
        logger.error(f"Error: Duration directory not found at {duration_dir}")
        logger.error("Run merge_parquet_files from duration_process.py to generate duration files.")
        exit(1)
    durations = glob.glob(os.path.join(duration_dir, "*.parquet"))
    if not durations:
        logger.error(f"Error: No duration data found in {duration_dir}")
        exit(1)
    user_profile_list = []
    for duration in durations:
        df = pd.read_parquet(duration, columns=["username", "profile_id"])
        user_profile_list.append(df.drop_duplicates())
    user_profile_df = pd.concat(user_profile_list, ignore_index=True).drop_duplicates()
    logger.info(f"Users before merge: {len(user_profile_df)}")
    user_profile_df = user_profile_df.merge(user_df, on="username", how="inner")
    load_time = time.time() - load_start
    logger.info(f"Data loaded in {load_time:.2f} seconds")
    logger.info(f"Loaded {len(user_profile_df)} users and {len(item_df)} items")
    logger.info(f"Unique profile_ids: {user_profile_df['profile_id'].nunique()}")
    logger.info(f"Unique items in item_df: {item_df['content_id'].nunique()}\n")
    if len(user_profile_df) == 0:
        logger.error("Error: user_profile_df is empty after merge")
        exit(1)

    # Load content metadata
    logger.info("=== Loading Content Metadata ===")
    meta_load_start = time.time()
    try:
        content_pl = pl.read_parquet(content_path)
    except Exception as e:
        logger.error(f"Error loading content metadata: {e}")
        exit(1)
    content_unique = (
        content_pl
        .unique(subset=['content_id'])
        .select(['content_id', 'content_name', 'tag_names', 'type_id'])
    )
    content_dict = {row[0]: (row[1], row[2], row[3]) for row in content_unique.iter_rows()}
    meta_load_time = time.time() - meta_load_start
    logger.info(f"Content metadata loaded in {meta_load_time:.2f} seconds\n")

    # Load model
    logger.info("=== Loading Model ===")
    model_load_start = time.time()
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        exit(1)
    expected_input_dim = checkpoint["model_state_dict"]["ECN.dfc.weight"].shape[1]
    model = DCNv3(expected_input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model_load_time = time.time() - model_load_start
    logger.info(f"Model loaded in {model_load_time:.2f} seconds\n")

    # Process users in chunks
    user_batch_size = 20
    num_users = len(user_profile_df)
    num_chunks = ceil(num_users / user_batch_size)
    total_pairs = 0
    chunk_times = []
    homepage_rule = []

    logger.info(f"=== Processing {num_users} Users in {num_chunks} Chunks ===")
    logger.info(f"num_users: {num_users}, num_chunks: {num_chunks}")
    if num_users == 0:
        logger.error("Error: No users to process")
        exit(1)
    # Clear result files if they exist
    if os.path.exists(result_jsonl_path):
        os.remove(result_jsonl_path)
    if os.path.exists(rule_content_path):
        os.remove(rule_content_path)

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
        if len(user_chunk) == 0:
            logger.error("Error: user_chunk is empty")
            continue

        # Perform cross join
        cross_start = time.time()
        user_chunk_pl = pl.from_pandas(user_chunk)
        item_pl = pl.from_pandas(item_df)
        cross_df = user_chunk_pl.join(item_pl, how="cross")
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

        # Save chunk results to final files
        save_start = time.time()
        save_chunk_results(result_dict, result_jsonl_path, rule_content_path, homepage_rule, content_dict, rule_info_path, tags_path, TOP_N)
        save_time = time.time() - save_start
        logger.info(f"  Save chunk results: {save_time:.2f} seconds")

        # Log memory usage
        import psutil
        logger.info(f"  Memory usage: {psutil.virtual_memory().percent}%\n")
        chunk_time = time.time() - chunk_start
        chunk_times.append(chunk_time)
        logger.info(f"  Total chunk time: {chunk_time:.2f} seconds | Pairs: {len(predictions):,}\n")

    # Convert result.jsonl to result.json in batches
    logger.info("=== Converting Result JSONL to JSON ===")
    convert_start = time.time()
    batch_size = 500
    final_result = {}
    if not os.path.exists(result_jsonl_path):
        logger.error("Error: result.jsonl not created, no results to convert")
        exit(1)
    with open(result_jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            final_result.update(data)
            if (i + 1) % batch_size == 0:
                with open(result_json_path + '.tmp', 'a', encoding='utf-8') as f_tmp:
                    json.dump(final_result, f_tmp, indent=4, ensure_ascii=False)
                final_result = {}
    if final_result:
        with open(result_json_path + '.tmp', 'a', encoding='utf-8') as f_tmp:
            json.dump(final_result, f_tmp, indent=4, ensure_ascii=False)
    if os.path.exists(result_json_path + '.tmp'):
        os.rename(result_json_path + '.tmp', result_json_path)
    convert_time = time.time() - convert_start
    logger.info(f"Result JSON converted in {convert_time:.2f} seconds\n")

    # Save rulename.json
    logger.info("=== Saving Rulename JSON ===")
    save_rulename_start = time.time()
    with open(rulename_json_path, 'w', encoding='utf-8') as f:
        json.dump(homepage_rule, f, indent=4, ensure_ascii=False)
    save_rulename_time = time.time() - save_rulename_start
    logger.info(f"Rulename JSON saved in {save_rulename_time:.2f} seconds\n")

    # Clean up temporary result.jsonl
    if os.path.exists(result_jsonl_path):
        os.remove(result_jsonl_path)
        logger.info(f"Temporary result.jsonl deleted")

    # Final timing summary
    total_time = time.time() - start_time
    logger.info("=== Final Timing Summary ===")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Total user-item pairs: {total_pairs:,}")
    logger.info(f"Average inference time per pair: {total_time / total_pairs:.6f} seconds" if total_pairs > 0 else "No pairs processed")
    logger.info(f"Average chunk time: {sum(chunk_times) / len(chunk_times):.2f} seconds" if chunk_times else "No chunks processed")
    logger.info(f"Number of chunks processed: {len(chunk_times)}")