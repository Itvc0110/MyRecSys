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

        #combined_df['label'] = (combined_df['percent_duration'] > 0.1).astype(int)
        #combined_df = combined_df.drop(columns=['percent_duration', 'duration'], inplace=False)
        #combined_df['content_duration'] = np.log(combined_df['content_duration'])

####################################################################################
        # compute watch counts using profile_id + content_id
        combined_df['watch_count'] = combined_df.groupby(['profile_id', 'content_id'])['content_id'].transform('count')

        # label: either watched more than 10% or more than 2 times
        combined_df['label'] = (
            (combined_df['percent_duration'] > 0.3) |
            (combined_df['watch_count'] >= 1)
        ).astype(int)

        combined_df = combined_df.drop(columns=['percent_duration', 'duration', 'watch_count'], inplace=False)
        combined_df['content_duration'] = np.log(combined_df['content_duration'])
####################################################################################

        combined_df.to_parquet(output_filepath, index=False)
        return combined_df
    else:
        return