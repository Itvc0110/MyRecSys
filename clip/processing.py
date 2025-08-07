from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from duration_process import merge_parquet_files
from item_process import process_clip_item
from user_process import process_user_data
import glob
from time import time
from math import ceil
import numpy as np
import os
from pathlib import Path
import pandas as pd
import polars as pl

def write_cross_chunk(args):
    i, user_chunk_pd, clip_df_pd, chunk_size, infer_subdir = args

    batch_start = time()
    user_chunk = pl.from_pandas(user_chunk_pd)
    clip_pl = pl.from_pandas(clip_df_pd)

    # Cross join
    cross_chunk = user_chunk.join(clip_pl, how="cross")

    # Save to parquet
    part_file = os.path.join(infer_subdir, f"infer_user_clip_part_{i}.parquet")
    io_start = time()
    cross_chunk.write_parquet(part_file)
    io_elapsed = time() - io_start
    batch_elapsed = time() - batch_start

    print(f"  ↪︎ Saved: {part_file} ({len(cross_chunk)} rows) "
          f"| Batch {batch_elapsed:.2f}s | I/O {io_elapsed:.2f}s")

    return len(cross_chunk)


def process_data(output_filepath):
    project_root = Path().resolve()

    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)

    duration_folder_path = "duration"
    duration_folder_path = os.path.join(project_root, duration_folder_path)

    merged_duration_folder_path = "clip/merged_duration"
    merged_duration_folder_path = os.path.join(project_root, merged_duration_folder_path)

    user_path = "month_mytv_info.parquet"
    user_path = os.path.join(project_root, user_path)

    clip_data_path = "mytv_vmp_content"
    clip_data_path = os.path.join(project_root, clip_data_path)

    durations = glob.glob(os.path.join(merged_duration_folder_path, "*.parquet"))
    if len(durations)<1:
        merge_parquet_files(duration_folder_path, merged_duration_folder_path)
        durations = glob.glob(os.path.join(merged_duration_folder_path, "*.parquet"))

    user_df = process_user_data(user_path, "clip/train_data", mode='train')
    clip_df = process_clip_item(clip_data_path, "clip/train_data", mode='train')
    clip_df['content_id'] = clip_df['content_id'].astype(str)

    all_merged_data = []

    for duration in durations:
        try:
            duration_df = pd.read_parquet(duration)
            duration_df['content_id'] = duration_df['content_id'].astype(str)

            print(f"\nProcessing {os.path.basename(duration)}")
            print(f"→ Duration rows: {len(duration_df)}")

            merged_with_user = pd.merge(duration_df, user_df, on='username', how='inner')
            print(f"→ After user merge: {len(merged_with_user)}")

            final_merged = pd.merge(merged_with_user, clip_df, on='content_id', how='inner')
            print(f"→ After clip merge: {len(final_merged)}")

            all_merged_data.append(final_merged)
        except Exception as e:
            print(f"Error processing {duration}: {str(e)}")

    if all_merged_data:
        combined_df = pd.concat(all_merged_data, ignore_index=True)
        print(f"\nTotal merged rows before drop duplicates: {len(combined_df)}")

        combined_df = combined_df.drop_duplicates()
        print(f"→ After drop_duplicates: {len(combined_df)}")

        combined_df['content_duration'] = combined_df['content_duration'].astype(float)
        combined_df['duration'] = combined_df['duration'].astype(float)
        combined_df['percent_duration'] = combined_df['duration']/combined_df['content_duration']
        combined_df['label'] = (combined_df['percent_duration'] > 0.3).astype(int)
        combined_df = combined_df.drop(columns=['percent_duration', 'duration'], inplace=False)
        combined_df['content_duration'] = np.log(combined_df['content_duration'])

        combined_df.to_parquet(output_filepath, index=False)
        return combined_df
    else:
        return


    
def process_infer_data(user_data_path, clip_data_path, num_user, num_clip, output_dir_path,
                       user_batch_size=20, chunk_size=None, max_files=-1):
    start_time = time()
    project_root = Path().resolve()
    output_dir = os.path.join(project_root, output_dir_path)
    os.makedirs(output_dir, exist_ok=True)

    infer_subdir = os.path.join(output_dir, "infer_user_clip")
    os.makedirs(infer_subdir, exist_ok=True)

    print("Loading user & clip data...")
    user_df = process_user_data(user_data_path, output_dir_path, num_user, mode='infer').head(num_user)
    clip_df = process_clip_item(clip_data_path, output_dir_path, num_clip, mode='infer').head(num_clip)
    clip_df['content_id'] = clip_df['content_id'].astype(str)

    if chunk_size is None:
        chunk_size = user_batch_size * len(clip_df)
    print(f"chunk_size set to {chunk_size} (user_batch_size × num_clips)")

    merged_duration_folder_path = os.path.join(project_root, "clip/merged_duration")
    durations = glob.glob(os.path.join(merged_duration_folder_path, "*.parquet"))
    user_profile_list = []
    for duration in durations:
        try:
            df = pd.read_parquet(duration, columns=["username", "profile_id"])
            user_profile_list.append(df.drop_duplicates())
        except Exception as e:
            print(f"Error reading {duration}: {e}")
    if not user_profile_list:
        print("No duration data available.")
        return

    user_profile_df = pd.concat(user_profile_list, ignore_index=True).drop_duplicates()
    user_profile_df = user_profile_df.merge(user_df, on="username", how="inner")

    total_users = len(user_profile_df)
    total_clips = len(clip_df)
    total_expected_rows = total_users * total_clips

    print(f"Loaded {total_users} unique user-profile entries.")
    print(f"Total clips: {total_clips}")
    print(f"Estimated total inference rows: {total_expected_rows:,}")

    user_profile_path = os.path.join(output_dir, "user_profile_data.parquet")
    user_profile_df.to_parquet(user_profile_path, index=False)

    cross_chunks = []

    for i in range(0, len(user_profile_df), user_batch_size):
        user_chunk = user_profile_df.iloc[i:i + user_batch_size]
        user_pl = pl.from_pandas(user_chunk)
        clip_pl = pl.from_pandas(clip_df)

        cross = user_pl.join(clip_pl, how="cross")
        cross_df = cross.to_pandas().fillna(0)

        cross_chunks.append(cross_df)

        print(f"Generated cross-chunk {i // user_batch_size + 1}: {len(cross_df)} rows")

    estimated_files = len(cross_chunks)
    print(f"{estimated_files} user chunks will be processed")

    elapsed = time() - start_time

    print("\nAll user batches proccessed.")
    print(f"Time taken to save all files: {elapsed:.2f} seconds")
    print(f"max_chunks={max_files} (currently unused in this mode)")