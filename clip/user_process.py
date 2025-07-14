import pandas as pd
import glob
import os
import joblib
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

ENC_DIR = "model/clip/encoder"
os.makedirs(ENC_DIR, exist_ok=True)

def process_user_data(data_path, output_dir, num_user = -1):
    user_data = pd.read_parquet(data_path)
    print("Available columns:", user_data.columns.tolist())

    user_data = user_data.dropna()
    user_data = user_data[user_data['birthday'].str.isdigit()]
    user_data['birthday'] = user_data['birthday'].str[:4].astype(int)
    user_data = user_data.drop(columns=["tentinh"])

    user_data['sex'] = user_data['sex'].astype(int)
    user_data['sex'] = user_data['sex'].apply(lambda x: x if x in [0, 1] else 1)

    cat_cols = ["province", "package_code"]
    enc_path = os.path.join(ENC_DIR, "user_ohe.joblib")
    user_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    user_ohe.fit(user_data[cat_cols])
    joblib.dump(user_ohe, enc_path)
    
    arr = user_ohe.transform(user_data[cat_cols])
    cols = user_ohe.get_feature_names_out(cat_cols)
    ohe_df = pd.DataFrame(arr, columns=cols, index=user_data.index)
    user_data = pd.concat([user_data.drop(cat_cols, axis=1), ohe_df], axis=1)

    project_root = Path().resolve()

    full_output_dir = os.path.join(project_root, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    user_data_path = f"{output_dir}/user_data.csv"
    user_data_path = os.path.join(project_root, user_data_path)
    if num_user == -1:
        user_data.to_csv(user_data_path, index=False)
    else:
        user_data = user_data.head(num_user)
        user_data.to_csv(user_data_path, index=False)
    return user_data




