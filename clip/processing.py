from pathlib import Path
from time import time
import os
import numpy as np
import pandas as pd
import polars as pl
import glob
from glob import glob  

from user_process import process_user_data
from item_process import process_clip_item
from duration_process import merge_parquet_files

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
                       user_batch_size=20):
    """
    Preprocess clip user & item data for inference.
    Ensures:
      - merged_content_clips.parquet exists with metadata
      - profile_id is merged into user data
    Yields (chunk_idx, user_chunk_df, full_clip_df) for on-the-fly inference.
    """
    overall_start = time()
    project_root = Path().resolve()
    output_dir = os.path.join(project_root, output_dir_path)
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 0: Ensure merged content metadata exists ---
    merged_content_path = os.path.join(output_dir, "merged_content_clips.parquet")
    if not os.path.exists(merged_content_path):
        print(f"[process_infer_data] Merged content metadata missing. Creating at {merged_content_path}...")
        clip_metadata_df = process_clip_item(clip_data_path, output_dir_path, num_clip, mode='infer')
        # Save only metadata columns if they exist
        meta_cols = ['content_id', 'content_name', 'tag_names', 'type_id']
        meta_cols = [c for c in meta_cols if c in clip_metadata_df.columns]
        if meta_cols:
            clip_metadata_df[meta_cols].drop_duplicates().to_parquet(merged_content_path, index=False)
            print(f"[process_infer_data] Saved metadata: {merged_content_path}")
        else:
            print(f"[Warning] No metadata columns found to save at {merged_content_path}")

    # --- Step 1: User preprocessing ---
    t0 = time()
    print("[process_infer_data] Loading & preprocessing user data...")
    user_df = process_user_data(user_data_path, output_dir_path, num_user, mode='infer').head(num_user)
    print(f"  ↪︎ User preprocess done: {len(user_df)} rows ({time() - t0:.2f}s)")

    # --- Step 2: Merge profile_id from duration data ---
    duration_dir = os.path.join(project_root, "clip/merged_duration")
    if not os.path.exists(duration_dir) or not glob.glob(os.path.join(duration_dir, "*.parquet")):
        print("[process_infer_data] Creating merged duration parquet files...")
        merge_parquet_files(os.path.join(project_root, "duration"), duration_dir)

    profile_map_list = []
    for f in glob.glob(os.path.join(duration_dir, "*.parquet")):
        try:
            df = pd.read_parquet(f, columns=["username", "profile_id"]).drop_duplicates()
            profile_map_list.append(df)
        except Exception as e:
            print(f"[Warning] Could not read {f}: {e}")

    if profile_map_list:
        profile_map_df = pd.concat(profile_map_list, ignore_index=True).drop_duplicates()
        user_df = user_df.merge(profile_map_df, on="username", how="left")

    if "profile_id" not in user_df.columns:
        raise RuntimeError("profile_id column could not be found/merged into user_df")

    # --- Step 3: Clip item preprocessing ---
    t0 = time()
    print("[process_infer_data] Loading & preprocessing clip item data...")
    clip_df = process_clip_item(clip_data_path, output_dir_path, num_clip, mode='infer').head(num_clip)
    clip_df['content_id'] = clip_df['content_id'].astype(str)
    print(f"  ↪︎ Clip preprocess done: {len(clip_df)} rows ({time() - t0:.2f}s)")

    # Save preprocessed data for reference
    user_profile_path = os.path.join(output_dir, "user_profile_data.parquet")
    clip_item_path = os.path.join(output_dir, "clip_item_data.parquet")
    user_df.to_parquet(user_profile_path, index=False)
    clip_df.to_parquet(clip_item_path, index=False)
    print(f"[process_infer_data] Saved user_profile to {user_profile_path}")
    print(f"[process_infer_data] Saved clip_item to {clip_item_path}")
    print(f"[process_infer_data] Total users: {len(user_df)}, Total clips: {len(clip_df)}, "
          f"Estimated rows: {len(user_df) * len(clip_df):,}")

    # --- Step 4: Yield chunks for on-the-fly inference ---
    total_users = len(user_df)
    for idx, start in enumerate(range(0, total_users, user_batch_size), start=1):
        yield idx, user_df.iloc[start:start + user_batch_size], clip_df

    print(f"[Total preprocessing time] {time() - overall_start:.2f}s")