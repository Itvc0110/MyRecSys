from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from duration_process import merge_parquet_files
from item_process import process_movie_item
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
    i, user_chunk_pd, movie_df_pd, chunk_size, infer_subdir = args

    batch_start = time()
    user_chunk = pl.from_pandas(user_chunk_pd)
    movie_pl = pl.from_pandas(movie_df_pd)

    # Cross join
    cross_chunk = user_chunk.join(movie_pl, how="cross")

    # Save to parquet
    part_file = os.path.join(infer_subdir, f"infer_user_movie_part_{i}.parquet")
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

    merged_duration_folder_path = "movie/merged_duration"
    merged_duration_folder_path = os.path.join(project_root, merged_duration_folder_path)

    user_path = "month_mytv_info.parquet"
    user_path = os.path.join(project_root, user_path)

    movie_data_path = "mytv_vmp_content"
    movie_data_path = os.path.join(project_root, movie_data_path)

    durations = glob.glob(os.path.join(merged_duration_folder_path, "*.parquet"))
    if len(durations)<1:
        merge_parquet_files(duration_folder_path, merged_duration_folder_path)
        durations = glob.glob(os.path.join(merged_duration_folder_path, "*.parquet"))

    user_df = process_user_data(user_path, "movie/train_data", mode='train')
    movie_df = process_movie_item(movie_data_path, "movie/train_data", mode='train')
    movie_df['content_id'] = movie_df['content_id'].astype(str)

    all_merged_data = []

    for duration in durations:
        try:
            duration_df = pd.read_parquet(duration)
            duration_df['content_id'] = duration_df['content_id'].astype(str)

            print(f"\nProcessing {os.path.basename(duration)}")
            print(f"→ Duration rows: {len(duration_df)}")

            merged_with_user = pd.merge(duration_df, user_df, on='username', how='inner')
            print(f"→ After user merge: {len(merged_with_user)}")

            final_merged = pd.merge(merged_with_user, movie_df, on='content_id', how='inner')
            print(f"→ After movie merge: {len(final_merged)}")

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
        combined_df['label'] = (combined_df['percent_duration'] > 0.05).astype(int)
        combined_df = combined_df.drop(columns=['percent_duration', 'duration'], inplace=False)
        combined_df['content_duration'] = np.log(combined_df['content_duration'])

        combined_df.to_parquet(output_filepath, index=False)
        return combined_df
    else:
        return


def process_infer_data(user_data_path, movie_data_path, num_user, num_movie, output_dir_path,
                       user_batch_size=20, chunk_size=None, max_files=-1):
    start_time = time()
    project_root = Path().resolve()
    output_dir = os.path.join(project_root, output_dir_path)
    os.makedirs(output_dir, exist_ok=True)

    infer_subdir = os.path.join(output_dir, "infer_user_movie")
    os.makedirs(infer_subdir, exist_ok=True)

    print("Loading user & movie data...")
    user_df = process_user_data(user_data_path, output_dir_path, num_user, mode='infer').head(num_user)
    movie_df = process_movie_item(movie_data_path, output_dir_path, num_movie, mode='infer').head(num_movie)
    movie_df['content_id'] = movie_df['content_id'].astype(str)

    if chunk_size is None:
        chunk_size = user_batch_size * len(movie_df)
    print(f"chunk_size set to {chunk_size} (user_batch_size × num_movies)")

    merged_duration_folder_path = os.path.join(project_root, "movie/merged_duration")
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
    total_movies = len(movie_df)
    total_expected_rows = total_users * total_movies

    print(f"Loaded {total_users} unique user-profile entries.")
    print(f"Total movies: {total_movies}")
    print(f"Estimated total inference rows: {total_expected_rows:,}")

    user_profile_path = os.path.join(output_dir, "user_profile_data.parquet")
    user_profile_df.to_parquet(user_profile_path, index=False)

    user_chunks = []
    for i in range(0, len(user_profile_df), user_batch_size):
        chunk = user_profile_df.iloc[i:i + user_batch_size]
        user_chunks.append((i, chunk, movie_df, chunk_size, infer_subdir))

    estimated_files = len(user_chunks)
    print(f"{estimated_files} user chunks will be processed in parallel.")
    print(f"→ Estimated number of output files: {estimated_files}")

    max_workers = cpu_count()
    print(f"Starting parallel processing with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        row_counts = list(executor.map(write_cross_chunk, user_chunks))

    elapsed = time() - start_time
    total_rows_written = sum(row_counts)

    print("\nAll user batches merged and saved.")
    print(f"Actual number of files saved: {len(row_counts)}")
    print(f"Actual total rows written: {total_rows_written:,}")
    print(f"Time taken to save all files: {elapsed:.2f} seconds")
    print(f"Estimated inference batches: {len(row_counts)}")
    print(f"max_files={max_files} (currently unused in this mode)")