# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import glob
import common

DATASET_PATH = "../dataset/"


def getstockinfo():
    # 過去3年分の「通期」の決算データを取得する
    # http://ke.kabupro.jp/down/20170719f.xls
    # ※エクセルシートに過去３年分のデータがあるが、最新の通期分以外は「情報公開日または更新日」が
    # 　正確でないため、20170719f,20160719f,20150719fの３ファイルを取得しそれぞれの最新の
    #   通期分データのみを取得。
    input_path = DATASET_PATH + "stock_info/*.csv"
    files = glob.glob(input_path)
    df_marge = pd.DataFrame()

    for file in files:
        df = pd.read_csv(file)
        df = \
            df[['証券コード', '企業名', '連結個別', '期首', '期末',
                '決算期間', '売上高', '営業利益', '経常利益', '純利益',
                '純資産又は株主資本', '総資産', '情報公開又は更新日']]
        df = df.loc[(df['連結個別']  == '連結') & (df['決算期間'] == '通期')]
        df = df.sort_values(['証券コード', '期首'])

        df = df.dropna()
        df = df.drop_duplicates(['証券コード']) # 最初の行以外は重複データを消す

        format = '%Y/%m/%d'
        serial_date1, serial_date2, serial_date3 = [], [], [] # 日付をシリアス値に変換
        for date1, date2, date3 in zip(df['期首'].values, df['期末'].values, df['情報公開又は更新日']):
            serial_date1.append(common.date_to_num(date1, format))
            serial_date2.append(common.date_to_num(date2, format))
            serial_date3.append(common.date_to_num(date3, format))
        for num1, num2, num3, num4, num5, num6 in \
            zip(df['売上高'].values, df['営業利益'].values, df['経常利益'].values, df['純利益'].values, df['純資産又は株主資本'].values, df['総資産'].values):
                num1_remvcom = int(str.replace(num1,',',''))
                num2_remvcom = int(str.replace(num2,',',''))
                num3_remvcom = int(str.replace(num3,',',''))
                num4_remvcom = int(str.replace(num4,',',''))
                num5_remvcom = int(str.replace(num5,',',''))
                num6_remvcom = int(str.replace(num6,',',''))
        df['期首'] = serial_date1
        df['期末'] = serial_date2
        df['情報公開又は更新日'] = serial_date3
        df['売上高'] = num1_remvcom
        df['営業利益'] = num2_remvcom
        df['経常利益'] = num3_remvcom
        df['純利益'] = num4_remvcom
        df['純資産又は株主資本'] = num5_remvcom
        df['総資産'] = num6_remvcom
        df_marge = pd.concat([df_marge, df], axis=0)

    df_marge = df.drop_duplicates(['証券コード','期首']) # 最初の行以外は重複データを消す
    df_marge = df_marge.drop(["企業名", "連結個別", "決算期間"], axis=1)
    df_stockinfo = df_marge
    df_stockinfo = df_stockinfo.sort_values(['証券コード', '期末'])

    return df_stockinfo

def getstocksplit():
    # 過去3年の株式分割データ/売買単位変更データを取得する。
    # https://mxp1.monex.co.jp/mst/servlet/ITS/info/StockSplit
    # http://kabu.com/investment/meigara/tani_henkou.html
    # より入手し、項目「権利付最終売買日」「銘柄コード」「分割比率」を加工。
    # （ホントはスクレイピングしたかった）
    input_path = DATASET_PATH + "stock_split/split.csv"
    df = pd.read_csv(input_path)
    serial_date1 = []
    for date1 in df['分割日'].values:
        format = '%Y/%m/%d'
        serial_date1.append(common.date_to_num(date1, format))
    df['分割日'] = serial_date1
    df = df.drop_duplicates(['分割日', '証券コード'])
    df.sort_values(by='分割日')

    return df

def run():
    df_stockinfo = getstockinfo()
    df_stocksplit = getstocksplit()

    print(df_stocksplit)

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
        df_marge = df_marge.drop_duplicates(['日付'])
        df_marge = df_marge.sort_values(by='日付')

        df_stocksplit_code = df_stocksplit.loc[(df_stocksplit['証券コード'] == int(code[0:4]))]
        df_stockinfo_code = df_stockinfo.loc[(df_stockinfo['証券コード'] == int(code[0:4]))]
        df_marge_stockinfo = pd.DataFrame()

        data_start, data_high, data_low, data_end, data_yield = \
            [], [], [], [], []
        for data in df_marge.values:
            # data: 0:日付 1:始値 2:高値 3:安値 4:終値 5:出来高
            if(np.isnan(data[1])):
                # 出来高がなかった場合
                start = start
                high = start
                low = start
                end = start
                yield_ = 0
            else:
                start = data[1]
                high = data[2]
                low = data[3]
                end = data[4]
                yield_ = data[5]
            for split_date in df_stocksplit_code['分割日'].values:
                if data[0] < split_date:
                    df_w = df_stocksplit_code.loc[(df_stocksplit_code['分割日'] == split_date)]
                    start = start * int(df_w['倍率'].values)
                    high = high * int(df_w['倍率'].values)
                    low = low * int(df_w['倍率'].values)
                    end = end * int(df_w['倍率'].values)
                    yield_ = yield_ * int(df_w['倍率'].values)
            data_start.append(start)
            data_high.append(high)
            data_low.append(low)
            data_end.append(end)
            data_yield.append(yield_)
        df_marge['始値'] = data_start
        df_marge['高値'] = data_high
        df_marge['安値'] = data_low
        df_marge['終値'] = data_end
        df_marge['出来高'] = data_yield

        for date_stock in df_marge['日付'].values:
            delete_flg = True
            df = df_stockinfo_code.sort_values(['期末'], ascending=False)

            for date_info in df['情報公開又は更新日'].values:
                if date_stock > date_info:
                    df_marge_stockinfo = pd.concat([df_marge_stockinfo, df_stockinfo_code.loc[(df_stockinfo_code["情報公開又は更新日"] == date_info)]], axis=0)
                    delete_flg = False
                    break
            if delete_flg:
                df_marge = df_marge[df_marge['日付'] != date_stock] # date_stockの行を削除

        df_marge = pd.concat([df_marge.reset_index(drop=True), df_marge_stockinfo.reset_index(drop=True)], axis=1)
        df_marge = df_marge.drop(["売買代金"], axis=1)

        # コード以外は常用対数
        if "証券コード" in df_marge.columns:
            df_code = df_marge["証券コード"]
            # df_marge = pd.np.log10(df_marge)
            df_marge = df_marge.replace([np.inf, -np.inf], np.nan).fillna(0)
            df_marge["証券コード"] = df_code
        df_marge.to_csv(output_path, index=False)
        print("complete! code:", code)

if __name__ == '__main__':
    run()
