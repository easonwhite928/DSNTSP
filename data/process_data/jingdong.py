import pandas as pd
from data.process_data.process_data import process_data

JING_DONG_PATH = 'data/jingdong/jingdong.csv'
JING_DONG_OUTPUT = 'jingdong'
JING_DONG_USER_COL = 'user_id'
JING_DONG_TIME_COL = 'day'
JING_DONG_ITEM_COL = 'sku_id'
JING_DONG_TIME_FORMAT = '%Y-%m-%d'
JING_DONG_MIN_ITEM_CNT = 3
JING_DONG_MIN_USER_ITEM_CNT = 3
JING_DONG_MAX_USER_ITEM_CNT = 10000
JING_DONG_MIN_USER_SET_CNT = 4
JING_DONG_MAX_USER_SET_CNT = 1000

def main():
    # df1 = pd.read_csv('data/jingdong/jd_201602.csv')
    # df2 = pd.read_csv('data/jingdong/jd_201603.csv')
    # df3 = pd.read_csv('data/jingdong/jd_201604.csv')
    # df = pd.concat([df1, df2, df3])
    df = pd.read_csv(JING_DONG_PATH)
    print(df['type'].value_counts())
    print(df.head())
    df = df[df['type'] == 2]
    df[JING_DONG_TIME_COL] = df['action_time'].map(lambda x: x.split()[0])
    df[JING_DONG_TIME_COL] = pd.to_datetime(df[JING_DONG_TIME_COL], format=JING_DONG_TIME_FORMAT)
    df[JING_DONG_TIME_COL] = df[JING_DONG_TIME_COL].map(lambda x: x.value // (24 * 60 * 60 * 10**9))
    process_data(df=df,
                 output=JING_DONG_OUTPUT,
                 user_col=JING_DONG_USER_COL,
                 time_col=JING_DONG_TIME_COL,
                 item_col=JING_DONG_ITEM_COL,
                 min_item_cnt=JING_DONG_MIN_ITEM_CNT,
                 min_user_item_cnt=JING_DONG_MIN_USER_ITEM_CNT,
                 max_user_item_cnt=JING_DONG_MAX_USER_ITEM_CNT,
                 min_user_set_cnt=JING_DONG_MIN_USER_SET_CNT,
                 max_user_set_cnt=JING_DONG_MAX_USER_SET_CNT)

if __name__ == '__main__':
    main()