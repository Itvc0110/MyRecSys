import pandas as pd
import glob
import json
import os
import joblib
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from pathlib import Path

def split_categories(x):
    if pd.isna(x):
        return []
    return str(x).split(',')

def onehot_encode(data):
    # Prepare encoder directory
    ENC_DIR = Path("model/clip/encoder")
    ENC_DIR.mkdir(parents=True, exist_ok=True)

    # 1) FIT & SAVE single‑valued encoder on full data
    single_cols = ["content_country","locked_level","VOD_CODE","contract","type_id"]
    single_path = ENC_DIR / "item_ohe_single.joblib"
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(data[single_cols])
    joblib.dump(ohe, single_path)

    # 2) FIT & SAVE multi‑valued encoder on full data
    mlb_path = ENC_DIR / "item_mlb_cate.joblib"
    mlb = MultiLabelBinarizer(sparse_output=False)
    lists = data["content_cate_id"].fillna("").apply(split_categories)
    mlb.fit(lists)
    joblib.dump(mlb, mlb_path)

    # 3) TRANSFORM both and drop originals
    ohe_arr = ohe.transform(data[single_cols])
    ohe_df  = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(single_cols),
                           index=data.index)

    mlb_arr = mlb.transform(lists)
    mlb_df  = pd.DataFrame(mlb_arr,
                           columns=[f"content_cate_id_{c}" for c in mlb.classes_],
                           index=data.index)

    data = data.drop(single_cols + ["content_cate_id"], axis=1)
    return pd.concat([data, ohe_df, mlb_df], axis=1)

def merge_content_clips(clip_data_path, output_file):
    # unchanged from original…
    clip_files = glob.glob(f'{clip_data_path}/**/movie_clip_*.json', recursive=True)
    all_data = []
    for f in clip_files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                js = json.load(fh)
                all_data.extend(js if isinstance(js, list) else [js])
        except:
            pass
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    return df

def fit_item_encoder(data, single_cols, mlb_col):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(data[single_cols])
    joblib.dump(ohe, Path("model/clip/encoder/item_ohe_single.joblib"))

    lists = data[mlb_col].fillna("").apply(split_categories)
    mlb = MultiLabelBinarizer(sparse_output=False)
    mlb.fit(lists)
    joblib.dump(mlb, Path("model/clip/encoder/item_mlb_cate.joblib"))

def transform_item_data(data, single_cols, mlb_col):
    ohe = joblib.load(Path("model/clip/encoder/item_ohe_single.joblib"))
    mlb = joblib.load(Path("model/clip/encoder/item_mlb_cate.joblib"))

    lists = data[mlb_col].fillna("").apply(split_categories)
    ohe_arr = ohe.transform(data[single_cols])
    mlb_arr = mlb.transform(lists)

    ohe_df = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(single_cols), index=data.index)
    mlb_df = pd.DataFrame(mlb_arr, columns=[f"content_cate_id_{c}" for c in mlb.classes_], index=data.index)
    data = data.drop(single_cols + [mlb_col], axis=1)
    return pd.concat([data, ohe_df, mlb_df], axis=1)

def process_clip_item(clip_data_path, output_dir, num_clip=-1, mode='train'):
    # Load or merge raw clip data
    project_root = Path().resolve()
    full_output_dir = project_root / output_dir
    full_output_dir.mkdir(parents=True, exist_ok=True)

    merged_file = full_output_dir / "merged_content_clips.csv"
    if not merged_file.exists():
        clip_df = merge_content_clips(clip_data_path, str(merged_file))
    else:
        dtype_spec = {
            'content_id': str,
            'content_single': str,
            'content_publish_year': 'float32',
            'content_country': str,
            'type_id': str,
            'tag_names': str,
            'content_duration': 'float32',
            'content_status': str,
            'locked_level': str,
            'contract': str,
            'VOD_CODE': str,
            'content_cate_id': str,
        }
        clip_df = pd.read_csv(merged_file, dtype=dtype_spec)

    # Clean durations
    clip_df['content_duration'] = pd.to_numeric(clip_df['content_duration'], errors='coerce')
    clip_df = clip_df[clip_df['content_duration'] > 0]

    # Keep only needed columns
    cols = ['content_id','content_single','content_publish_year','content_country',
            'type_id','tag_names','content_duration','content_status',
            'locked_level','contract','VOD_CODE','content_cate_id']
    clip_df = clip_df[cols]

    # Encoder setup
    single_cols = ["content_country", "locked_level", "VOD_CODE", "contract", "type_id"]
    mlb_col = "content_cate_id"

    # Train mode: fit and save encoder
    if mode == 'train':
        fit_item_encoder(clip_df, single_cols, mlb_col)

    # Transform with saved encoder
    clip_df = transform_item_data(clip_df, single_cols, mlb_col)

    # Slice clip data if needed
    if num_clip != -1:
        clip_df = (clip_df[(clip_df['content_status'] == "1") & (clip_df['tag_names'].str.contains(r'\w', na=False))]
                    .head(num_clip)
                    .dropna(subset=['tag_names']))

    if 'tag_names' in clip_df.columns:
        clip_df = clip_df.drop('tag_names', axis=1)
    # Save final output
    clip_df.to_csv(full_output_dir / "clip_item_data.csv", index=False)

    return clip_df