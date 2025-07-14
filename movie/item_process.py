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
    ENC_DIR = Path("model/movie/encoder")
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

def merge_content_movies(movie_data_path, output_file):
    # unchanged from original…
    movie_files = glob.glob(f'{movie_data_path}/**/content_movie_*.json', recursive=True)
    all_data = []
    for f in movie_files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                js = json.load(fh)
                all_data.extend(js if isinstance(js, list) else [js])
        except:
            pass
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    return df

def process_movie_item(movie_data_path, output_dir, num_movie=-1):
    # 1) Load (or merge) raw movies
    project_root = Path().resolve()
    full_output_dir = project_root / output_dir
    full_output_dir.mkdir(parents=True, exist_ok=True)

    merged_file = full_output_dir / "merged_content_movies.csv"
    if not merged_file.exists():
        movie_df = merge_content_movies(movie_data_path, str(merged_file))
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
        movie_df = pd.read_csv(merged_file, dtype=dtype_spec)

    # 2) Handle bad durations (logging + smoothing)
    movie_df['content_duration'] = pd.to_numeric(movie_df['content_duration'], errors='coerce')

    # Count categories
    num_lt_0 = (movie_df['content_duration'] < 0).sum()
    num_eq_0 = (movie_df['content_duration'] == 0).sum()
    num_gt_0 = (movie_df['content_duration'] > 0).sum()

    print(f"[INFO] content_duration < 0: {num_lt_0}")
    print(f"[INFO] content_duration == 0: {num_eq_0}")
    print(f"[INFO] content_duration > 0: {num_gt_0}")

    # Smooth non-positive durations using epsilon
    epsilon = 1.0
    movie_df['content_duration'] = movie_df['content_duration'].apply(
        lambda x: x if x > 0 else epsilon
        )

    # 3) Keep only needed cols
    cols = ['content_id','content_single','content_publish_year','content_country',
            'type_id','tag_names','content_duration','content_status',
            'locked_level','contract','VOD_CODE','content_cate_id']
    movie_df = movie_df[cols]

    # 4) One‑hot + multi‑label encode (always fit+save)
    movie_enc = onehot_encode(movie_df).drop_duplicates()

    # 5) Slice if needed & save
    out = movie_enc if num_movie==-1 else (
        movie_enc[movie_enc['content_status']=="1"]
               .head(num_movie)
               .dropna(subset=['tag_names'])
               .drop('tag_names', axis=1)
    )
    out.to_csv(full_output_dir/"movie_item_data.csv", index=False)

    return movie_enc
