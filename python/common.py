import codecs
import datetime as dt
import pandas as pd
import urllib.request
import time

DATASET_PATH = "../dataset/"

def sjis_to_utf8(input_path, output_path):
    '''
    SJIS -> UTF-8 にコード変換。
        input_path : コード変換するファイルのパス
        output_path : コード変換したファイルを生成するパス
    '''
    input = codecs.open(input_path, "r", "shift_jis")
    output = codecs.open(output_path, "w", "utf-8")

    for line in input:
        output.write(line)
    input.close()
    output.close()

def date_to_num(date, format):
    '''
    date型 -> int型(シリアル値)変換 ※Excelのシリアル値とは一致しない(らしい)
        引数
            date : date型の配列
            format : 変換する日付の形状 (例: '%Y/%m/%d')
        返り値
            num : dateをシリアル値に変換(1990/1/1をゼロとする)
    '''

    date = dt.datetime.strptime(date, format)
    date_zero = dt.datetime.strptime('1900/1/1', format)
    timedelta = date - date_zero
    num = timedelta.days
    return num

def get_stockscode():
    '''
    現存する株のコードの配列を返す
        事前準備
            http://k-db.com/stocks/yyyy-mm-dd?download=csv
            上記リンクより最新日の株式データを取得し、dataset配下に「stocks_code.csv」と
            リネームして格納しておくこと
        引数 なし
        返り値
            証券コードの配列
    '''
    file_path_code_sjis = DATASET_PATH + 'stocks_code.csv'
    file_path_code_utf8 = DATASET_PATH + 'stocks_code_utf8.csv'

    sjis_to_utf8(file_path_code_sjis, file_path_code_utf8)
    df = pd.read_csv(file_path_code_utf8)

    return df['コード'].values

class TimeMeasure:
    def __init__(self):
        self.start_ = time.time()

    def get(self):
        return time.time() - self.start_

    def reset(self):
        self.start_ = time.time()
