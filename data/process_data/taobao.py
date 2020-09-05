import pandas as pd
from data.process_data.process_data import process_data

OPTION = 'buy'  # buy | click

TAO_BAO_PATH = 'data/taobao/taobao15.csv'
if OPTION == 'buy':
    TAO_BAO_OUTPUT = 'taobao_buy'
    TAO_BAO_ACT_ID = 1
else:
    TAO_BAO_OUTPUT = 'taobao_click'
    TAO_BAO_ACT_ID = 0
TAO_BAO_USER_COL = 'use_ID'
TAO_BAO_TIME_COL = 'time'
TAO_BAO_ITEM_COL = 'ite_ID'
TAO_BAO_ITEM_TYPE_COL = 'cat_ID'
TAO_BAO_TIME_FORMAT = '%Y%m%d'
TAO_BAO_MIN_ITEM_CNT = 30
TAO_BAO_MIN_USER_ITEM_CNT = 10
TAO_BAO_MAX_USER_ITEM_CNT = 1000
TAO_BAO_MIN_USER_SET_CNT = 4
TAO_BAO_MAX_USER_SET_CNT = 100

def main():
    df = pd.read_csv(TAO_BAO_PATH)
    if OPTION == 'buy':
        # df = df[(df['time'] >= 20150901) & (df['time'] <= 20150931) & (df['act_ID'] == TAO_BAO_ACT_ID)]
        df = df[(df['time'] >= 20150801) & (df['time'] <= 20150915) & (df['act_ID'] == TAO_BAO_ACT_ID)]
    else:
        df = df[(df['time'] >= 20150701) & (df['time'] <= 20150721) & (df['act_ID'] == TAO_BAO_ACT_ID)]
    print(df.head())
    df[TAO_BAO_TIME_COL] = pd.to_datetime(df[TAO_BAO_TIME_COL], format=TAO_BAO_TIME_FORMAT)
    df[TAO_BAO_TIME_COL] = df[TAO_BAO_TIME_COL].map(lambda x: x.value // (24 * 60 * 60 * 10 ** 9))
    process_data(df=df,
                 output=TAO_BAO_OUTPUT,
                 user_col=TAO_BAO_USER_COL,
                 time_col=TAO_BAO_TIME_COL,
                 item_col=TAO_BAO_ITEM_COL,
                 cat_col=TAO_BAO_ITEM_TYPE_COL,
                 min_item_cnt=TAO_BAO_MIN_ITEM_CNT,
                 min_user_item_cnt=TAO_BAO_MIN_USER_ITEM_CNT,
                 max_user_item_cnt=TAO_BAO_MAX_USER_ITEM_CNT,
                 min_user_set_cnt=TAO_BAO_MIN_USER_SET_CNT,
                 max_user_set_cnt=TAO_BAO_MAX_USER_SET_CNT)

if __name__ == '__main__':
    main()
