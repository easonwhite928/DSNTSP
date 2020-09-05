import json
from pathlib import Path

from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def process_data(df,
                 output,
                 user_col,
                 time_col,
                 item_col,
                 cat_col,
                 min_item_cnt = 10,
                 min_user_item_cnt=10,
                 max_user_item_cnt=1000,
                 min_user_set_cnt=4,
                 max_user_set_cnt=100):
    df = df.copy()

    # remove items that purchased less than min_item_cnt
    print('remove items that purchased less than min_item_cnt ...')
    item_counts = df[item_col].value_counts()
    item_counts = item_counts[item_counts >= min_item_cnt]
    df = df[df[item_col].isin(item_counts.index)]

    # remove user that purchased less than min_user_item_cnt items and purchased less than min_user_set_cnt set
    print('remove user that purchased less than min_user_item_cnt items and purchased less than min_user_set_cnt set ...')
    user_item_counts = df[user_col].value_counts()
    user_set_counts = df.groupby(user_col)[time_col].nunique()
    user_item_counts = user_item_counts[(user_item_counts >= min_user_item_cnt) & (user_item_counts <= max_user_item_cnt)]
    user_set_counts = user_set_counts[(user_set_counts >= min_user_set_cnt) & (user_set_counts <= max_user_set_cnt)]
    df = df[df[user_col].isin(user_item_counts.index) & df[user_col].isin(user_set_counts.index)]

    # ordinal encoding
    print('ordinal encoding ...')
    item_ordinal_encoder = OrdinalEncoder()
    user_ordinal_encoder = OrdinalEncoder()
    cat_ordinal_encoder = OrdinalEncoder()
    df['item_ordinal'] = item_ordinal_encoder.fit_transform(df[item_col].values.reshape(-1, 1))
    df['user_ordinal'] = user_ordinal_encoder.fit_transform(df[user_col].values.reshape(-1, 1))
    df['cat_ordinal'] = cat_ordinal_encoder.fit_transform(df[cat_col].values.reshape(-1, 1))
    df['item_ordinal'] = df['item_ordinal'].astype(int)
    df['user_ordinal'] = df['user_ordinal'].astype(int)
    df['cat_ordinal'] = df['cat_ordinal'].astype(int)
    assert len(df['item_ordinal'].unique()) == df['item_ordinal'].max() - df['item_ordinal'].min() + 1
    assert len(df['user_ordinal'].unique()) == df['user_ordinal'].max() - df['user_ordinal'].min() + 1

    # generate item and category
    item_cat_df = df[['item_ordinal', 'cat_ordinal']]

    # generate data info
    data_info = {
        'num_users': df[user_col].nunique(),
        'num_items': df[item_col].nunique(),
        'num_sets': len(df.groupby([user_col, time_col])),
        'num_set_items_mean': df.groupby([user_col, time_col])[item_col].count().mean(),
        'num_set_items_min': df.groupby([user_col, time_col])[item_col].count().min(),
        'num_set_items_max': df.groupby([user_col, time_col])[item_col].count().max(),
        'num_user_sets_mean': df.groupby(user_col)[time_col].nunique().mean(),
        'num_user_sets_min': df.groupby(user_col)[time_col].nunique().min(),
        'num_user_sets_max': df.groupby(user_col)[time_col].nunique().max(),
        'num_user_items_mean': df.groupby(user_col)[item_col].count().mean(),
        'num_user_items_min': df.groupby(user_col)[item_col].count().min(),
        'num_user_items_max': df.groupby(user_col)[item_col].count().max(),
    }
    print(data_info)

    # generate data
    print('generate data ...')
    data = []
    user_grouped = df.groupby('user_ordinal')
    for user_id, user_group in tqdm(user_grouped):
        user = {
            'user_id': user_id,
            'sets': []
        }
        user_group = user_group.sort_values(by=time_col, ascending=True)
        user_set_grouped = user_group.groupby(time_col)
        for time, user_set_group in user_set_grouped:
            user['sets'].append({
                # 'timestamp': time.value // (24 * 60 * 60 * 10**9),  # timestamp day
                'timestamp': time,
                'items': user_set_group['item_ordinal'].tolist()
            })
        data.append(user)

    output_folder = Path('../data')
    data_path = output_folder / f'{output}.json'
    data_info_path = output_folder / f'{output}_info.json'

    with open(data_path, 'w') as f:
        f.write(json.dumps(data))

    with open(data_info_path, 'w') as f:
        f.write(json.dumps(data_info))
