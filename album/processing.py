from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from duration_process import merge_parquet_files
from item_process import process_album_item
from user_process import process_user_data
import glob
from time import time
from math import ceil
import numpy as np
import os
from pathlib import Path
import pandas as pd
import polars as pl

def process_data(output_filepath):
    project_root = Path().resolve()

    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)

    duration_folder_path = "duration"
    duration_folder_path = os.path.join(project_root, duration_folder_path)

    merged_duration_folder_path = "album/merged_duration"
    merged_duration_folder_path = os.path.join(project_root, merged_duration_folder_path)

    user_path = "month_mytv_info.parquet"
    user_path = os.path.join(project_root, user_path)

    album_data_path = "mytv_vmp_content"
    album_data_path = os.path.join(project_root, album_data_path)

    durations = glob.glob(os.path.join(merged_duration_folder_path, "*.parquet"))
    if len(durations)<1:
        merge_parquet_files(duration_folder_path, merged_duration_folder_path)
        durations = glob.glob(os.path.join(merged_duration_folder_path, "*.parquet"))

    user_df = process_user_data(user_path, "album/train_data", mode='train')
    album_df = process_album_item(album_data_path, "album/train_data", mode='train')
    album_df['content_id'] = album_df['content_id'].astype(str)

    all_merged_data = []

    for duration in durations:
        try:
            duration_df = pd.read_parquet(duration)
            duration_df['content_id'] = duration_df['content_id'].astype(str)

            print(f"\nProcessing {os.path.basename(duration)}")
            print(f"→ Duration rows: {len(duration_df)}")
            print(f"   Unique duration content_id: {duration_df['content_id'].nunique()}")
            print(f"   Sample duration content_id: {duration_df['content_id'].head(5).tolist()}")

            print(f"User DF rows: {len(user_df)}, unique usernames: {user_df['username'].nunique()}")
            print(f"Album DF rows: {len(album_df)}, unique content_id: {album_df['content_id'].nunique()}")
            print(f"   Sample album content_id: {album_df['content_id'].head(5).tolist()}")

            merged_with_user = pd.merge(duration_df, user_df, on='username', how='inner')
            print(f"→ After user merge: {len(merged_with_user)}")
            print(f"   Unique usernames after merge: {merged_with_user['username'].nunique()}")

            final_merged = pd.merge(merged_with_user, album_df, on='content_id', how='inner')
            print(f"→ After album merge: {len(final_merged)}")
            print(f"   Unique content_id after merge: {final_merged['content_id'].nunique()}")

            # Optional: check overlap explicitly
            overlap = set(duration_df['content_id']).intersection(set(album_df['content_id']))
            print(f"   Overlap content_id count: {len(overlap)}")
            if overlap:
                print(f"   Sample overlap content_id: {list(overlap)[:5]}")

            all_merged_data.append(final_merged)

        except Exception as e:
            print(f"Error processing {duration}: {str(e)}")

    if all_merged_data:
        combined_df = pd.concat(all_merged_data, ignore_index=True)
        print(f"\nTotal merged rows before drop duplicates: {len(combined_df)}")

        combined_df = combined_df.drop_duplicates()
        print(f"→ After drop_duplicates: {len(combined_df)}")

        #combined_df['content_duration'] = combined_df['content_duration'].astype(float)
        combined_df['duration'] = combined_df['duration'].astype(float)
        #combined_df['percent_duration'] = combined_df['duration']/combined_df['content_duration']

        combined_df['watch_count'] = combined_df.groupby(['profile_id', 'content_id'])['content_id'].transform('count')

        combined_df['label'] = (
            #(combined_df['percent_duration'] >= 0.3) |
            (combined_df['watch_count'] >= 2)
        ).astype(int)

        combined_df = combined_df.drop(columns=['duration', 'watch_count'], inplace=False)
        #combined_df['content_duration'] = np.log(combined_df['content_duration'])

        combined_df.to_parquet(output_filepath, index=False)
        return combined_df
    else:
        return
    
def process_infer_data(processed_user_path, user_data_path, processed_item_path, album_data_path, content_album_path, num_user=-1, num_album=-1):

    print("Preprocessing user and item data...")
    preprocess_start = time.time()
    if not os.path.exists(processed_user_path):
        print("  ↪︎ Processing user data...")
        process_user_data(user_data_path, output_dir="album/infer_data", num_user=num_user, mode='infer')
    else:
        print("  ↪︎ User data already exists, skipping preprocessing.")
    if not os.path.exists(processed_item_path):
        print("  ↪︎ Processing item data...")
        process_album_item(album_data_path, output_dir="album/infer_data", num_album=num_album, mode='infer')
    else:
        print("  ↪︎ Item data already exists, skipping preprocessing.")
    print(f"Preprocessing completed in {time.time()-preprocess_start:.2f} seconds")

    print("\nLoading user and item data...")
    user_df = pd.read_parquet(processed_user_path)
    album_df = pd.read_parquet(processed_item_path)

    project_root = Path().resolve()
    duration_dir = os.path.join(project_root, "album/merged_duration")
    durations = glob.glob(os.path.join(duration_dir, "*.parquet"))
    user_profile_list = []
    for duration in durations:
        df = pd.read_parquet(duration, columns=["username", "profile_id"])
        user_profile_list.append(df.drop_duplicates())
    user_profile_df = pd.concat(user_profile_list, ignore_index=True).drop_duplicates()
    user_profile_df = user_profile_df.merge(user_df, on="username", how="inner")
    print(f"Data loaded in {time.time()-preprocess_start:.2f} seconds")

    album_pl = pl.from_pandas(album_df)
    total_contents = album_pl.height
    print(f"\nTotal unique contents: {total_contents:,}")

    content_album_pl = pl.read_parquet(content_album_path)
    content_unique = (
        content_album_pl
        .unique(subset=['content_id'])
        .select(['content_id', 'content_name', 'tag_names', 'type_id'])
    )
    content_dict = {row[0]: (row[1], row[2], row[3]) for row in content_unique.iter_rows()}
    return user_profile_df, album_pl, content_dict