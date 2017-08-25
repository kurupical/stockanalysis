import codecs
import datetime as dt
import pandas as pd
import numpy as np
import urllib.request
import time
import datetime
import os
from configparser import *


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
    date_zero = dt.datetime.strptime('1900/1/1', "%Y/%m/%d")
    timedelta = date - date_zero
    num = timedelta.days
    return num

def num_to_date(num, format):

    date = dt.datetime(1899, 12, 31) + dt.timedelta(days=num)
    return date.strftime(format)

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

def str_to_list(str, split_char):
    '''
    文字列strを文字split_charごとに切り分けた配列にして返す
    '''
    return [x for x in str.split(split_char) if len(x) != 0]

def make_log(log_path, log_header=None, config_path="work_config/outputlog.ini"):
    '''
    ログ出力先を設定し、ログファイルを作成する。
    ログファイルのヘッダに、指定した文字列(log_header)を出力する。
    '''
    # configファイルの作成
    config = ConfigParser()
    config['param'] = { 'log_path':  log_path }
    with open(config_path, 'w') as configfile:
        config.write(configfile)

    # logファイルの作成/書き込み
    if os.path.exists(log_path):
        file = open(log_path, 'a')
    else:
        file = open(log_path, 'w')

    now = datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")

    file.write("==========================================================\n")
    file.write("start:" + str(now) + "\n")
    file.write(log_header + "\n")
    file.write("==========================================================\n")
    file.close()

def output_log(log, object=None, config_path="work_config/outputlog.ini"):
    '''
    ログを出力する。
    　time[class]:log
    '''

    # make_logconfで作成したconfigファイルの読み込み
    config = ConfigParser()
    config.read(config_path)
    log_path = config['param']['log_path']

    # ログ出力内容の編集
    now = datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    if object is not None:
        object = object.__class__.__name__

    # ログ出力
    file = open(log_path, 'a')
    file.write(str(now) +  " [" + str(object) + "] " + log + "\n")
    file.close()

class TimeMeasure:
    def __init__(self):
        self.start_ = time.time()

    def get(self):
        return time.time() - self.start_

    def reset(self):
        self.start_ = time.time()

class StdConverter:
    '''
        データ群の標準化、標準化戻しを行う
        attribute
            numpy/pandas data : 標準化対象のデータ
            pandas _std_info : dataの配列ごとの平均、標準偏差を格納
    '''
    def __init__(self, data):
        self.data = data
        self.data_type = type(data)

        self.std(self.data)

    def std(self, data=None):
        self.data_std = data
        self._std_info = pd.DataFrame()

        # numpy/pandas
        if type(data) == type(pd.DataFrame()):
            obj_type = "pd"
            self.length = len(data.columns)
            data = data.values
        elif type(data) == type(np.array([[]])):
            obj_type = "np"
            self.length = len(data[0])
            data = data

        for i in range(self.length):
            ary = np.copy(data[:,i])
            ary_std = (ary - ary.mean()) / ary.std()
            if obj_type == "pd":
                self.data_std.iloc[:, i] = ary_std
            elif obj_type == "np":
                self.data_std[:, i] = ary_std
            std_info = pd.Series([i, ary.mean(), ary.std()], index=['item' ,'mean' ,'std'])
            self._std_info = self._std_info.append(std_info, ignore_index=True)

    def unstd(self, data=None):
        # numpy/pandas
        if type(data) == type(pd.DataFrame()):
            data = data.values
        elif type(data) == type(np.array([[]])):
            data = data

        data_unstd = np.copy(data)
        for i in range(self.length):
            df = self._std_info[self._std_info["item"] == i]
            data_unstd = data_unstd * df['std'].values + df['mean'].values
        return data_unstd
