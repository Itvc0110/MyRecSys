import os
import glob
import time
from pathlib import Path
import pandas as pd
import polars as pl

from duration_process import merge_parquet_files
from item_process import process_clip_item
from user_process import process_user_data


def process_data(output_filepath):
    project_root = Path().resolve()
    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)

    duration_folder_path = project_root / "duration"
    merged_duration_folder_path = project_root / "clip/merged_duration"
    user_path = project_root / "month_mytv_info.parquet"
    clip_data_path = project_root / "mytv_vmp_content"

    durations = glob.glob(str(merged_duration_folder_path / "*.parquet"))
    if len(durations) < 1:
        merge_parquet_files(duration_folder_path, merged_duration_folder_path)
        durations = glob.glob(str(merged_duration_folder_path / "*.parquet"))

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
        combined_df['percent_duration'] = combined_df['duration'] / combined_df['content_duration']
        combined_df['label'] = (combined_df['percent_duration'] > 0.3).astype(int)
        combined_df = combined_df.drop(columns=['percent_duration', 'duration'], inplace=False)

        combined_df.to_parquet(output_filepath, index=False)
        return combined_df
    else:
        return


def generate_user_item_stream(user_data_path, item_data_path, num_user=-1, num_item=-1, 
                              user_batch_size=50, mode='infer'):
    """
    Yield (user_chunk, cross_join_chunk) batches as Polars DataFrames without writing parquet.
    """
    project_root = Path().resolve()
    print("Loading user & item data...")
    
    # Load users
    user_df = process_user_data(user_data_path, "clip/infer_data", num_user, mode=mode)
    # Load items (clips or movies)
    item_df = process_clip_item(item_data_path, "clip/infer_data", num_item, mode=mode)
    item_df['content_id'] = item_df['content_id'].astype(str)
    
    total_users = len(user_df)
    total_items = len(item_df)
    print(f"Loaded {total_users} users × {total_items} items = {total_users * total_items:,} pairs (streamed)")

    # Convert items to Polars for efficient cross join
    item_pl = pl.from_pandas(item_df)

    # Stream user batches
    for start in range(0, len(user_df), user_batch_size):
        user_chunk_pd = user_df.iloc[start:start + user_batch_size]
        user_chunk_pl = pl.from_pandas(user_chunk_pd)

        # Cross join: Each user batch with all items
        cross_chunk = user_chunk_pl.join(item_pl, how="cross")
        yield user_chunk_pd, cross_chunk
