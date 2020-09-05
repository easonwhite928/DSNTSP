import pandas as pd
from data.process_data.process_data import process_data

TA_FENG_PATH = 'data/tafeng/tafeng.csv'
TA_FENG_OUTPUT = 'tafeng'
TA_FENG_USER_COL = 'CUSTOMER_ID'
TA_FENG_TIME_COL = 'TRANSACTION_DT'
TA_FENG_ITEM_COL = 'PRODUCT_ID'
TA_FENG_ITEM_TYPE_COL = 'PRODUCT_SUBCLASS'
TA_FENG_TIME_FORMAT = '%m/%d/%Y'
TA_FENG_MIN_ITEM_CNT = 10
TA_FENG_MIN_USER_ITEM_CNT = 10
TA_FENG_MAX_USER_ITEM_CNT = 1000
TA_FENG_MIN_USER_SET_CNT = 4
TA_FENG_MAX_USER_SET_CNT = 100

def main():
    df = pd.read_csv(TA_FENG_PATH)
    print(df.head())
    df[TA_FENG_TIME_COL] = pd.to_datetime(df[TA_FENG_TIME_COL], format=TA_FENG_TIME_FORMAT)
    df[TA_FENG_TIME_COL] = df[TA_FENG_TIME_COL].map(lambda x: x.value // (24 * 60 * 60 * 10 ** 9))
    process_data(df=df,
                 output=TA_FENG_OUTPUT,
                 user_col=TA_FENG_USER_COL,
                 time_col=TA_FENG_TIME_COL,
                 item_col=TA_FENG_ITEM_COL,
                 min_item_cnt=TA_FENG_MIN_ITEM_CNT,
                 min_user_item_cnt=TA_FENG_MIN_USER_ITEM_CNT,
                 max_user_item_cnt=TA_FENG_MAX_USER_ITEM_CNT,
                 min_user_set_cnt=TA_FENG_MIN_USER_SET_CNT,
                 max_user_set_cnt=TA_FENG_MAX_USER_SET_CNT)

if __name__ == '__main__':
    main()
