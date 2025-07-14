import os
import glob
import pandas as pd

def merge_parquet_files(duration_folder_path, merged_duration_folder_path):
    if not os.path.isdir(merged_duration_folder_path):
        os.makedirs(merged_duration_folder_path)
    
    parquet_files = glob.glob(os.path.join(duration_folder_path, "*.parquet"))
    print(parquet_files)
    dfs = []
    for file in parquet_files:
        try:
            columns_to_keep = ['username', 'content_id', 'duration', 'profile_id']
            df = pd.read_parquet(file)
            df = df[columns_to_keep]
            dfs.append(df)
        except Exception:
            pass
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        
        total_rows = len(combined_df)
        
        rows_per_file = total_rows // 5
        remainder = total_rows % 5
        
        for i in range(5):
            start_idx = i * rows_per_file
            end_idx = start_idx + rows_per_file + (remainder if i == 4 else 0)
            
            df_slice = combined_df.iloc[start_idx:end_idx]
            
            output_file = os.path.join(merged_duration_folder_path, f"merged_part_{i+1}.csv")
            print(f"Saving {len(df_slice)} rows to {output_file}")
            df_slice.to_csv(output_file, index=False)
        