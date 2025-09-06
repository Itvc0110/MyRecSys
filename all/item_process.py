import pandas as pd
import glob
import json
import os
import joblib
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from pathlib import Path

ENC_DIR = Path("model/all/encoder")
ENC_DIR.mkdir(parents=True, exist_ok=True)

def split_categories(x):
    if pd.isna(x):
        return []
    return str(x).split(',')

def merge_content_alls(all_data_path, output_file):

    patterns = ["content_movie_2*.json", "content_clip_2*.json", "music_video_2*.json"]
    
    all_files = []

    for p in patterns:
        all_files.extend(glob.glob(f"{all_data_path}/**/{p}", recursive=True))

    all_data = []
    for f in all_files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                js = json.load(fh)
                all_data.extend(js if isinstance(js, list) else [js])
        except:
            pass
        
    df = pd.DataFrame(all_data)
    df.to_parquet(output_file, index=False)
    return df

def fit_item_encoder(data, single_cols, mlb_col):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(data[single_cols])
    joblib.dump(ohe, Path("model/all/encoder/item_ohe_single.joblib"))

    lists = data[mlb_col].fillna("").apply(split_categories)
    mlb = MultiLabelBinarizer(sparse_output=False)
    mlb.fit(lists)
    joblib.dump(mlb, Path("model/all/encoder/item_mlb_cate.joblib"))

def transform_item_data(data, single_cols, mlb_col):
    ohe = joblib.load(Path("model/all/encoder/item_ohe_single.joblib"))
    mlb = joblib.load(Path("model/all/encoder/item_mlb_cate.joblib"))
    scaler = joblib.load(ENC_DIR / "item_scaler.joblib")

    ohe_arr = ohe.transform(data[single_cols])
    ohe_df = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(single_cols), index=data.index)

    mlb_lists = data[mlb_col].fillna("").apply(split_categories)
    mlb_arr = mlb.transform(mlb_lists)
    mlb_df = pd.DataFrame(mlb_arr, columns=[f"content_cate_id_{c}" for c in mlb.classes_], index=data.index)

    data = data.drop(single_cols + [mlb_col], axis=1)
    return pd.concat([data, ohe_df, mlb_df], axis=1)

def process_all_item(all_data_path, output_dir, num_all=-1, mode='train'):
    project_root = Path().resolve()
    full_output_dir = project_root / output_dir
    full_output_dir.mkdir(parents=True, exist_ok=True)

    merged_file = full_output_dir / "merged_content_alls.parquet"
    if not merged_file.exists():
        all_df = merge_content_alls(all_data_path, str(merged_file))
    else:
        dtype_spec = {
            "content_id": str,
            "type_id": str,
            "tag_names": str,
            "content_duration": "float32",
            "content_status": str,
            "VOD_CODE": str,
            "content_cate_id": str,
        }
        all_df = pd.read_parquet(merged_file, dtype=dtype_spec)

    # Clean durations
    all_df['content_duration'] = pd.to_numeric(all_df['content_duration'], errors='coerce')
    all_df = all_df[all_df['content_duration'] > 0]

    cols = ['content_id',
            'type_id','tag_names','content_duration',
            'content_status',
            'VOD_CODE','content_cate_id']
    all_df = all_df[cols]

    # encoder
    single_cols = ["VOD_CODE", "type_id"]
    mlb_col = "content_cate_id"

    # train mode
    if mode == 'train':
        fit_item_encoder(all_df, single_cols, mlb_col)

    # transform
    all_df = transform_item_data(all_df, single_cols, mlb_col)

    # filter with content_status and drop those with not suitable tag_names
    if num_all!= -1:
        all_df = (all_df[(all_df['content_status'] == "1") & (all_df['tag_names'].str.contains(r'\w', na=False))]
                    .head(num_all)
                    .dropna(subset=['tag_names']))

    if 'tag_names' in all_df.columns:
        all_df = all_df.drop('tag_names', axis=1)
    # save final output
    all_df.to_parquet(full_output_dir / "all_item_data.parquet", index=False)

    return all_df