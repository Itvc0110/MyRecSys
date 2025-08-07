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

def process_infer_data(user_data_path, clip_data_path, num_user, num_clip,
                       user_batch_size=20):

    overall_start = time()
    print("[Start] process_infer_data")

    project_root = Path().resolve()

    t0 = time()
    clip_df = process_clip_item(clip_data_path, None, num_clip, mode='infer')
    clip_df['content_id'] = clip_df['content_id'].astype(str)
    clip_pl = pl.from_pandas(clip_df)
    print(f"[Time] Loaded clip data in {time() - t0:.2f}s")

    t1 = time()
    user_df = process_user_data(user_data_path, None, num_user, mode='infer')
    user_df['username'] = user_df['username'].astype(str)
    print(f"[Time] Loaded user data in {time() - t1:.2f}s")

    t2 = time()
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
        raise RuntimeError("No duration data available.")

    user_profile_df = pd.concat(user_profile_list, ignore_index=True).drop_duplicates()
    print(f"[Time] Loaded profile info in {time() - t2:.2f}s")

    t3 = time()
    user_profile_df = user_profile_df.merge(user_df, on="username", how="inner")
    print(f"[Time] Merged user profiles in {time() - t3:.2f}s")

    total_users = len(user_profile_df)
    print(f"Total users: {total_users}, Total clips: {len(clip_df)}")

    for i in range(0, total_users, user_batch_size):
        user_chunk = user_profile_df.iloc[i:i + user_batch_size]
        user_pl = pl.from_pandas(user_chunk)

        t_batch = time()
        cross = user_pl.join(clip_pl, how="cross")
        yield cross.to_pandas().fillna(0)
        print(f"[Time] Generated cross-chunk {i // user_batch_size + 1} in {time() - t_batch:.2f}s")

    print(f"[End] process_infer_data completed in {time() - overall_start:.2f}s")