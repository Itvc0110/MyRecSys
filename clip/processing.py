from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from duration_process import merge_parquet_files
from item_process import process_clip_item
from user_process import process_user_data
import glob, os
from time import time
import numpy as np
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

# processing.py  -- replace the existing process_infer_data with this

def process_infer_data(user_data_path, movie_data_path, num_user, num_movie, output_dir_path,
                       user_batch_size=20, max_rows_warn_threshold=5_000_000):
    """
    Preprocess user and movie data, save to two parquet files, and yield user chunks
    together with full movie dataframe for on-the-fly inference.

    Parameters:
      user_batch_size: number of user-profile rows to process per chunk
      max_rows_warn_threshold: warn if chunk_size * n_movies exceeds this (protect memory)
    Yields:
      (chunk_idx (1-based), user_chunk_df (pandas), movie_df (pandas))
    """
    from time import time
    project_root = Path().resolve()
    output_dir = os.path.join(project_root, output_dir_path)
    os.makedirs(output_dir, exist_ok=True)

    overall_start = time()

    # 1) User preprocessing
    t0 = time()
    print("[process_infer_data] Loading & preprocessing user data...")
    user_df = process_user_data(user_data_path, output_dir_path, num_user, mode='infer').head(num_user)
    user_elapsed = time() - t0
    print(f"  ↪︎ User preprocess done: {len(user_df)} rows ({user_elapsed:.2f}s)")

    # 2) Item preprocessing
    t0 = time()
    print("[process_infer_data] Loading & preprocessing movie item data...")
    movie_df = process_movie_item(movie_data_path, output_dir_path, num_movie, mode='infer').head(num_movie)
    movie_df['content_id'] = movie_df['content_id'].astype(str)
    movie_elapsed = time() - t0
    print(f"  ↪︎ Movie preprocess done: {len(movie_df)} rows ({movie_elapsed:.2f}s)")

    # 3) Save the two canonical files once
    user_profile_path = os.path.join(output_dir, "user_profile_data.parquet")
    movie_item_path = os.path.join(output_dir, "movie_item_data.parquet")
    user_df.to_parquet(user_profile_path, index=False)
    movie_df.to_parquet(movie_item_path, index=False)
    print(f"[process_infer_data] Saved user_profile to {user_profile_path}")
    print(f"[process_infer_data] Saved movie_item to {movie_item_path}")

    total_users = len(user_df)
    total_movies = len(movie_df)
    est_total_pairs = total_users * total_movies
    print(f"[process_infer_data] Total users: {total_users}, Total movies: {total_movies}, Estimated rows: {est_total_pairs:,}")

    # Safety advice / warn if chunk × movies huge
    est_per_chunk = user_batch_size * total_movies
    if est_per_chunk > max_rows_warn_threshold:
        print(f"WARNING: user_batch_size * total_movies = {est_per_chunk:,} > {max_rows_warn_threshold:,}.")
        print("  → Reduce user_batch_size or consider splitting movies into sub-batches.")

    # 4) Yield chunks for on-the-fly processing
    chunk_idx = 0
    for start in range(0, total_users, user_batch_size):
        chunk_idx += 1
        user_chunk = user_df.iloc[start:start + user_batch_size].copy()
        yield chunk_idx, user_chunk, movie_df

    print(f"[process_infer_data] Total preprocessing time: {time() - overall_start:.2f}s")
