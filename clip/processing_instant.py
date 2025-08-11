# Add or replace in processing.py

import os
import glob
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from time import time
import pandas as pd
import polars as pl

from duration_process import merge_parquet_files
from item_process import process_clip_item
from user_process import process_user_data

def _write_cross_chunk(args):
    """
    Worker that receives a tuple (i, user_chunk_pd, clip_df_pd, infer_subdir)
    and writes a parquet file named infer_user_clip_part_{i}.parquet
    Returns the written file path and number of rows.
    """
    i, user_chunk_pd, clip_df_pd, infer_subdir = args
    batch_start = time()

    user_chunk = pl.from_pandas(user_chunk_pd)
    clip_pl = pl.from_pandas(clip_df_pd)
    cross_chunk = user_chunk.join(clip_pl, how="cross")

    part_file = os.path.join(infer_subdir, f"infer_user_clip_part_{i}.parquet")
    io_start = time()
    cross_chunk.write_parquet(part_file)
    io_elapsed = time() - io_start
    batch_elapsed = time() - batch_start

    print(f"  ↪︎ Saved: {part_file} ({len(cross_chunk)} rows) | Batch {batch_elapsed:.2f}s | I/O {io_elapsed:.2f}s")
    return part_file, len(cross_chunk)


def process_infer_data(user_data, clip_data,
                       num_user=-1, num_clip=-1,
                       output_dir_path="clip/infer_data",
                       user_batch_size=20, chunk_size=None,
                       max_workers=None,
                       force_rebuild=False):
    
    project_root = os.path.abspath(os.getcwd())
    output_dir = os.path.join(project_root, output_dir_path)
    infer_subdir = os.path.join(output_dir, "infer_user_clip")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(infer_subdir, exist_ok=True)

    # If parts exist and not forcing rebuild, return them
    existing_parts = sorted(glob.glob(os.path.join(infer_subdir, "infer_user_clip_part_*.parquet")))
    if existing_parts and not force_rebuild:
        print(f"Found {len(existing_parts)} prebuilt infer parts in {infer_subdir}, skipping build.")
        # ensure user_profile_data exists
        user_profile_path = os.path.join(output_dir, "user_profile_data.parquet")
        return {
            "infer_files": existing_parts,
            "user_profile_path": user_profile_path,
            "total_users": None,
            "total_clips": None
        }

    # 1) Prepare user_profile_df
    # If user_data is a DataFrame, assume it's the processed user_df (from process_user_data)
    if isinstance(user_data, pd.DataFrame):
        user_df = user_data
    else:
        # user_data is path to raw user parquet
        user_df = process_user_data(user_data, output_dir_path, num_user=num_user, mode='infer')

    # 2) Prepare clip_df
    if isinstance(clip_data, pd.DataFrame):
        clip_df = clip_data
    else:
        # clip_data is path (folder) to raw clip content
        clip_df = process_clip_item(clip_data, output_dir_path, num_clip=num_clip, mode='infer')

    # ensure content_id as str
    clip_df['content_id'] = clip_df['content_id'].astype(str)

    # 3) Build user_profile_df from merged_duration files (same logic as infer.py)
    merged_duration_folder_path = os.path.join(project_root, "clip/merged_duration")
    if not os.path.exists(merged_duration_folder_path):
        # fallback: try to create merged durations if available
        print("Merged duration folder not found. Please run duration merge step first.")
        raise FileNotFoundError(f"{merged_duration_folder_path} not found")

    duration_files = sorted(glob.glob(os.path.join(merged_duration_folder_path, "*.parquet")))
    if not duration_files:
        raise FileNotFoundError(f"No parquet files in {merged_duration_folder_path}")

    user_profile_list = []
    for duration in duration_files:
        try:
            df = pd.read_parquet(duration, columns=["username", "profile_id"])
            user_profile_list.append(df.drop_duplicates())
        except Exception as e:
            print(f"Warning: failed to read {duration}: {e}")

    if not user_profile_list:
        raise RuntimeError("No duration user-profile data available")

    user_profile_df = pd.concat(user_profile_list, ignore_index=True).drop_duplicates()
    user_profile_df = user_profile_df.merge(user_df, on="username", how="inner")

    total_users = len(user_profile_df)
    total_clips = len(clip_df)
    total_expected_rows = total_users * total_clips
    print(f"Loaded {total_users} unique user-profile entries.")
    print(f"Total clips: {total_clips}")
    print(f"Estimated total inference rows: {total_expected_rows:,}")

    # Save user_profile_df for later reference
    user_profile_path = os.path.join(output_dir, "user_profile_data.parquet")
    user_profile_df.to_parquet(user_profile_path, index=False)

    # 4) split into chunks and write parts in parallel
    user_chunks = []
    idx = 0
    for i in range(0, len(user_profile_df), user_batch_size):
        chunk_pd = user_profile_df.iloc[i:i + user_batch_size]
        user_chunks.append((idx, chunk_pd, clip_df, infer_subdir))
        idx += 1

    estimated_files = len(user_chunks)
    print(f"{estimated_files} user chunks will be produced.")

    max_workers = max_workers or cpu_count()
    print(f"Starting parallel cross-join with {max_workers} workers...")

    written = []
    with ProcessPoolExecutor(max_workers=max_workers) as exec:
        for part_file, row_count in exec.map(_write_cross_chunk, user_chunks):
            written.append((part_file, row_count))

    written_sorted = sorted(written, key=lambda x: x[0])
    infer_files = [p for p, _ in written_sorted]

    print(f"\nAll user batches merged and saved. Files: {len(infer_files)} | Total rows written: {sum(r for _, r in written_sorted):,}")

    return {
        "infer_files": infer_files,
        "user_profile_path": user_profile_path,
        "total_users": total_users,
        "total_clips": total_clips
    }
