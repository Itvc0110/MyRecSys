from duration_process import merge_parquet_files
from item_process import process_other_item
from user_process import process_user_data
import glob
import os
from pathlib import Path
import pandas as pd

def process_data(output_filepath):
    project_root = Path().resolve()

    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)

    duration_folder_path = "duration"
    duration_folder_path = os.path.join(project_root, duration_folder_path)

    merged_duration_folder_path = "other/merged_duration"
    merged_duration_folder_path = os.path.join(project_root, merged_duration_folder_path)

    user_path = "month_mytv_info.parquet"
    user_path = os.path.join(project_root, user_path)

    other_data_path = "mytv_vmp_content"
    other_data_path = os.path.join(project_root, other_data_path)

    durations = glob.glob(os.path.join(merged_duration_folder_path, "*.csv"))
    if len(durations)<1:
        merge_parquet_files(duration_folder_path, merged_duration_folder_path)
        durations = glob.glob(os.path.join(merged_duration_folder_path, "*.csv"))

    user_df = process_user_data(user_path, "other/train_data", mode='train')
    other_df = process_other_item(other_data_path, "other/train_data", mode='train')
    other_df['content_id'] = other_df['content_id'].astype(str)

    all_merged_data = []

    for duration in durations:
        try:
            duration_df = pd.read_csv(duration)
            duration_df['content_id'] = duration_df['content_id'].astype(str)

            print(f"\nProcessing {os.path.basename(duration)}")
            print(f"→ Duration rows: {len(duration_df)}")

            merged_with_user = pd.merge(duration_df, user_df, on='username', how='inner')
            print(f"→ After user merge: {len(merged_with_user)}")

            final_merged = pd.merge(merged_with_user, other_df, on='content_id', how='inner')
            print(f"→ After other merge: {len(final_merged)}")

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
        
        combined_df.to_csv(output_filepath, index=False)
        return combined_df
    else:
        return


def process_infer_data(user_data_path, other_data_path, num_user, num_other ,output_filepath):
    project_root = Path().resolve()
    output_filepath = os.path.join(project_root, output_filepath)

    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)

    user_data_path = os.path.join(project_root, user_data_path)
    other_data_path = os.path.join(project_root, other_data_path)

    user_df = process_user_data(user_data_path, "other/infer_data", num_user, mode='infer')
    other_df = process_other_item(other_data_path, "other/infer_data", num_other, mode='infer')
    user_df = user_df.head(num_user)
    other_df = other_df.head(num_other)
    other_df['content_id'] = other_df['content_id'].astype(str)

    merged_duration_folder_path = "other/merged_duration"
    merged_duration_folder_path = os.path.join(project_root, merged_duration_folder_path)
    durations = glob.glob(os.path.join(merged_duration_folder_path, "*.csv"))
    user_profile_list = []
    for duration in durations:
        try:
            duration_df = pd.read_csv(duration)
            unique_pairs_df = duration_df[['username', 'profile_id']].drop_duplicates()
            user_profile_list.append(unique_pairs_df)
        except Exception as e:
            print(f"Error processing {duration}: {str(e)}")
    
    if user_profile_list:
        user_profile_df = pd.concat(user_profile_list, ignore_index=True)
        user_profile_df = user_profile_df.drop_duplicates()

    user_profile_df = user_profile_df.merge(user_df, on='username', how='inner')

    user_profile_path = os.path.join(project_root, "other/infer_data/user_profile_data.csv")
    os.makedirs(os.path.dirname(user_profile_path), exist_ok=True)  # Also ensure user profile dir
    user_profile_df.to_csv(user_profile_path, index=False)

    user_other_df = user_profile_df.merge(other_df, how='cross')
    user_other_df.to_csv(output_filepath, index=False)

    return user_other_df