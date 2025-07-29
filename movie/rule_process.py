from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import glob
import os

def _assign_rule_user(args):
    user_id, suggested_content, rule_dict, tag_mapping = args
    for content_id, content in suggested_content.items():
        content['rule_id'] = process_rule(rule_dict, tag_mapping, content['tag_names'])
    return user_id, suggested_content

def get_tag_mapping(tags_path):
    content_to_tag_file = glob.glob(f"{tags_path}/*/*.json")
    dfs = []
    for item in content_to_tag_file:
        columns_to_keep = ['TAG_ID', 'TAG_NAME']
        df = pd.read_json(item)
        df = df[columns_to_keep]
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(inplace=False)
    combined_df.columns = combined_df.columns.str.lower()
    combined_df = combined_df.reset_index(drop=True)
    combined_df.to_parquet('movie/infer_data/tag_mapping.parquet', index=False)

    tag_mapping = {}
    for _, mapping in combined_df.iterrows():
            tag_mapping[mapping['tag_name']] = mapping['tag_id']
    return tag_mapping

def preprocess_rule_info(rule_df):
    rule_dict = {}
    for _, rule in rule_df.iterrows():
        rule_dict[rule['LIST_TAGS_ID']] = rule['RULE_ID']
    return rule_dict

def process_rule(rule_dict, tag_mapping, tag_names):
    tag_names_set = set(tag_names.split(', '))
    tag_ids = set(str(tag_mapping.get(tag_name)) for tag_name in tag_names_set)
    for tag_set, rule_id in rule_dict.items():
        tag_set = set(tag_set.split(','))
        if tag_set.issubset(tag_ids):
            return rule_id
    return -100

def get_rulename_parallel(reordered_data, rule_info_path, tags_path, max_workers=None):
    rule_info = pd.read_parquet(rule_info_path)
    rule_dict = preprocess_rule_info(rule_info)
    tag_mapping = get_tag_mapping(tags_path)

    args_list = [
        (uid, data['suggested_content'], rule_dict, tag_mapping)
        for uid, data in reordered_data.items()
    ]

    with ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
        results = executor.map(_assign_rule_user, args_list)

    return {uid: {'suggested_content': sc, 'user': reordered_data[uid]['user']} for uid, sc in results}