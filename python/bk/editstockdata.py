# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import glob
import MySQLdb
import common
import pandas.io.sql as psql

DATASET_PATH = "../dataset/"


df = pd.read_csv(DATASET_PATH + "stock_info.csv")
df_stockinfo = \
    df[['証券コード', '企業名', '連結個別', '期首', '期末',
        '決算期間', '売上高', '営業利益', '経常利益', '純利益',
        '純資産又は株主資本', '総資産']]

format = '%Y/%m/%d'
df_stockinfo = df_stockinfo.loc[(df['連結個別'] == '連結') & (df['決算期間'] == '通期')]
serial_date1, serial_date2 = [], [] # 日付をシリアス値に変換
for date1, date2 in zip(df_stockinfo['期首'].values, df_stockinfo['期末'].values):
    serial_date1.append(common.date_to_num(date1, format))
    serial_date2.append(common.date_to_num(date2, format))
df_stockinfo['期首'] = serial_date1
df_stockinfo['期末'] = serial_date2
df_stockinfo = df_stockinfo.dropna()
df_stockinfo = df_stockinfo.sort_values(['証券コード', '期末'])
df_stockinfo = df_stockinfo.drop_duplicates(['証券コード','期末'])


# output_path = DATASET_PATH + "stock_analysis/test.csv"
print(df_stockinfo.info())
# df_stockinfo.to_csv(output_path)

codes = common.get_stockscode()

# 銘柄ごとにデータをマージ(株式情報もくっつける)
for code in codes:
    input_path = DATASET_PATH + "stock_work/" + code + "201[5-7]_utf8.csv" #3年分を分析
    output_path = DATASET_PATH + "stock_analysis/" + code + ".csv"

    files = glob.glob(input_path)
    df_marge = pd.DataFrame()

    for file in files:
        try:
            df = pd.read_csv(file)
            df_marge = pd.concat([df, df_marge], axis=0)
        except:
            print("error:", sys.exc_info()[0], " file:", file)
    serial_date = []
    format = '%Y-%m-%d'
    for date in df_marge['日付'].values:
        serial_date.append(common.date_to_num(date, format))
    df_marge['日付'] = serial_date
    df_marge = df_marge.sort_values(by='日付')
    print(df_marge['日付'])

    df_stockinfo_code = df_stockinfo.loc[(df_stockinfo['証券コード'] == int(code[0:4]))]
    print(df_stockinfo_code)

    df_marge.to_csv(output_path, index=False)
    break
# データ編集
