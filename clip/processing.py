import os
import glob
import time
from pathlib import Path
import pandas as pd
import polars as pl

from duration_process import merge_parquet_files
from item_process import process_movie_item
from user_process import process_user_data


def process_data(output_filepath):
    project_root = Path().resolve()
    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)

    duration_folder_path = project_root / "duration"
    merged_duration_folder_path = project_root / "movie/merged_duration"
    user_path = project_root / "month_mytv_info.parquet"
    movie_data_path = project_root / "mytv_vmp_content"

    durations = glob.glob(str(merged_duration_folder_path / "*.parquet"))
    if len(durations) < 1:
        merge_parquet_files(duration_folder_path, merged_duration_folder_path)
        durations = glob.glob(str(merged_duration_folder_path / "*.parquet"))

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
        combined_df['percent_duration'] = combined_df['duration'] / combined_df['content_duration']
        combined_df['label'] = (combined_df['percent_duration'] > 0.3).astype(int)
        combined_df = combined_df.drop(columns=['percent_duration', 'duration'], inplace=False)

        combined_df.to_parquet(output_filepath, index=False)
        return combined_df
    else:
        return


def generate_user_chunks(user_data_path, movie_data_path, num_user, num_movie, user_batch_size=50):
    """
    Instead of saving parquet files, yield each user batch cross-joined with movies.
    """
    project_root = Path().resolve()

    print("Loading user & movie data...")
    user_df = process_user_data(user_data_path, "", num_user, mode='infer').head(num_user)
    movie_df = process_movie_item(movie_data_path, "", num_movie, mode='infer').head(num_movie)
    movie_df['content_id'] = movie_df['content_id'].astype(str)

    # Build user profile list
    merged_duration_folder_path = project_root / "movie/merged_duration"
    durations = glob.glob(str(merged_duration_folder_path / "*.parquet"))
    user_profile_list = [
        pd.read_parquet(d, columns=["username", "profile_id"]).drop_duplicates()
        for d in durations
    ]
    user_profile_df = pd.concat(user_profile_list, ignore_index=True).drop_duplicates()
    user_profile_df = user_profile_df.merge(user_df, on="username", how="inner")

    total_users = len(user_profile_df)
    total_movies = len(movie_df)
    total_expected_rows = total_users * total_movies

    print(f"Loaded {total_users} unique user-profile entries.")
    print(f"Total movies: {total_movies}")
    print(f"Estimated total inference rows: {total_expected_rows:,}")

    for start_idx in range(0, total_users, user_batch_size):
        batch_users = user_profile_df.iloc[start_idx:start_idx + user_batch_size]
        yield start_idx, batch_users, movie_df
