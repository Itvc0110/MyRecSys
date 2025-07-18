import pandas as pd
import glob
import os
import joblib
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

ENC_DIR = "model/movie/encoder"
os.makedirs(ENC_DIR, exist_ok=True)

def fit_user_encoder(user_data, cat_cols):
    user_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    user_ohe.fit(user_data[cat_cols])

    enc_path = os.path.join(ENC_DIR, "user_ohe.joblib")
    joblib.dump(user_ohe, enc_path)  
    return user_ohe

def transform_user_data(user_data, cat_cols):
    enc_path = os.path.join(ENC_DIR, "user_ohe.joblib")
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"Missing encoder at: {enc_path}")
    
    user_ohe = joblib.load(enc_path)
    arr = user_ohe.transform(user_data[cat_cols])
    cols = user_ohe.get_feature_names_out(cat_cols)
    ohe_df = pd.DataFrame(arr, columns=cols, index=user_data.index)
    return pd.concat([user_data.drop(cat_cols, axis=1), ohe_df], axis=1)


def process_user_data(data_path, output_dir, num_user=-1, mode='train'):
    user_data = pd.read_parquet(data_path)

    user_data = user_data.dropna()
    user_data = user_data[user_data['birthday'].str.isdigit()]
    user_data['birthday'] = user_data['birthday'].str[:4].astype(int)
    user_data = user_data.drop(columns=["tentinh"])
    user_data['sex'] = user_data['sex'].astype(int)
    user_data['sex'] = user_data['sex'].apply(lambda x: x if x in [0, 1] else 1)

    cat_cols = ["province", "package_code"]

    if mode == 'train':
        fit_user_encoder(user_data, cat_cols)
    user_data = transform_user_data(user_data, cat_cols)

    if num_user != -1:
        user_data = user_data.head(num_user)

    user_data_path = os.path.join(Path().resolve(), output_dir, "user_data.parquet")
    os.makedirs(os.path.dirname(user_data_path), exist_ok=True)
    user_data.to_parquet(user_data_path, index=False)

    return user_data