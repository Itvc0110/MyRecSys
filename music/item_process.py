import pandas as pd
import glob
import json
import os
import joblib
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, StandardScaler
from pathlib import Path

ENC_DIR = Path("model/music/encoder")
ENC_DIR.mkdir(parents=True, exist_ok=True)

def split_categories(x):
    if pd.isna(x):
        return []
    return str(x).split(',')

def merge_content_musics(music_data_path, output_file):
    music_files = glob.glob(f'{music_data_path}/**/music_video_2*.json', recursive=True)
    all_data = []
    for f in music_files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                js = json.load(fh)
                all_data.extend(js if isinstance(js, list) else [js])
        except:
            pass
    df = pd.DataFrame(all_data)
    df.to_parquet(output_file, index=False)
    return df

def fit_item_encoder(data, single_cols, mlb_col, cont_cols):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(data[single_cols])
    joblib.dump(ohe, Path("model/music/encoder/item_ohe_single.joblib"))

    lists = data[mlb_col].fillna("").apply(split_categories)
    mlb = MultiLabelBinarizer(sparse_output=False)
    mlb.fit(lists)
    joblib.dump(mlb, Path("model/music/encoder/item_mlb_cate.joblib"))

    scaler = StandardScaler()
    scaler.fit(data[cont_cols])
    joblib.dump(scaler, Path("model/music/encoder/item_scaler.joblib"))

def transform_item_data(data, single_cols, mlb_col, cont_cols):
    ohe = joblib.load(Path("model/music/encoder/item_ohe_single.joblib"))
    mlb = joblib.load(Path("model/music/encoder/item_mlb_cate.joblib"))
    scaler = joblib.load(ENC_DIR / "item_scaler.joblib")

    ohe_arr = ohe.transform(data[single_cols])
    ohe_df = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(single_cols), index=data.index)

    mlb_lists = data[mlb_col].fillna("").apply(split_categories)
    mlb_arr = mlb.transform(mlb_lists)
    mlb_df = pd.DataFrame(mlb_arr, columns=[f"content_cate_id_{c}" for c in mlb.classes_], index=data.index)

    scaled_arr = scaler.transform(data[cont_cols])
    scaled_df = pd.DataFrame(scaled_arr, columns=cont_cols, index=data.index)

    data = data.drop(single_cols + [mlb_col] + cont_cols, axis=1)
    return pd.concat([data, ohe_df, mlb_df, scaled_df], axis=1)

def process_music_item(music_data_path, output_dir, num_music=-1, mode='train'):
    project_root = Path().resolve()
    full_output_dir = project_root / output_dir
    full_output_dir.mkdir(parents=True, exist_ok=True)

    merged_file = full_output_dir / "merged_content_musics.parquet"
    if not merged_file.exists():
        music_df = merge_content_musics(music_data_path, str(merged_file))
    else:
        dtype_spec = {
            'content_id': str,
            'content_publish_year': 'float32',
            # 'content_country': str,
            'type_id': str,
            'tag_names': str,
            'content_duration': 'float32',
            'content_status': str,
            'VOD_CODE': str,
            'content_cate_id': str,
        }
        music_df = pd.read_parquet(merged_file, dtype=dtype_spec)

    # Clean durations
    music_df['content_duration'] = pd.to_numeric(music_df['content_duration'], errors='coerce')
    music_df = music_df[music_df['content_duration'] > 0]

    music_df["content_publish_year"] = pd.to_numeric(music_df["content_publish_year"].astype(str).str[:4], errors='coerce')
    music_df["content_publish_year"] = music_df["content_publish_year"].fillna(music_df["content_publish_year"].mean())

    cols = ['content_id','content_publish_year', 
            #'content_country',
            'type_id','tag_names','content_duration','content_status',
            'VOD_CODE','content_cate_id']
    music_df = music_df[cols]

    # encoder
    single_cols = [#"content_country",
                   "VOD_CODE", "type_id"]
    mlb_col = "content_cate_id"
    cont_cols = ["content_publish_year"]

    # train mode
    if mode == 'train':
        fit_item_encoder(music_df, single_cols, mlb_col, cont_cols)

    # transform
    music_df = transform_item_data(music_df, single_cols, mlb_col, cont_cols)

    # filter with content_status and drop those with not suitable tag_names
    if num_music != -1:
        music_df = (music_df[(music_df['content_status'] == "1") & (music_df['tag_names'].str.contains(r'\w', na=False))]
                    .head(num_music)
                    .dropna(subset=['tag_names']))

    if 'tag_names' in music_df.columns:
        music_df = music_df.drop('tag_names', axis=1)
    # save final output
    music_df.to_parquet(full_output_dir / "music_item_data.parquet", index=False)

    return music_df