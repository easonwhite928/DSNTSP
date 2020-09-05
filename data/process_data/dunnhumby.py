import pandas as pd
from data.process_data.process_data import process_data

DUNNHUMBY_PATH = 'data/dunnhumby/dunnhumby.csv'
DUNNHUMBY_OUTPUT = 'dunnhumby_all'
DUNNHUMBY_USER_COL = 'household_key'
DUNNHUMBY_TIME_COL = 'DAY'
DUNNHUMBY_ITEM_COL = 'PRODUCT_ID'
DUNNHUMBY_MIN_ITEM_CNT = 10
DUNNHUMBY_MIN_USER_ITEM_CNT = 10
DUNNHUMBY_MAX_USER_ITEM_CNT = 10000
DUNNHUMBY_MIN_USER_SET_CNT = 4
DUNNHUMBY_MAX_USER_SET_CNT = 1000

def main():
    df = pd.read_csv(DUNNHUMBY_PATH)
    print(df.head())
    df = df[(df.DAY >= 1) & (df.DAY <= 300)]
    process_data(df=df,
                 output=DUNNHUMBY_OUTPUT,
                 user_col=DUNNHUMBY_USER_COL,
                 time_col=DUNNHUMBY_TIME_COL,
                 item_col=DUNNHUMBY_ITEM_COL,
                 min_item_cnt=DUNNHUMBY_MIN_ITEM_CNT,
                 min_user_item_cnt=DUNNHUMBY_MIN_USER_ITEM_CNT,
                 max_user_item_cnt=DUNNHUMBY_MAX_USER_ITEM_CNT,
                 min_user_set_cnt=DUNNHUMBY_MIN_USER_SET_CNT,
                 max_user_set_cnt=DUNNHUMBY_MAX_USER_SET_CNT)

if __name__ == '__main__':
    main()