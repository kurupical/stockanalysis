# -*- coding: utf-8 -*-

import pandas as pd
import os
import time
import sys
import codecs
import common

BASE_URL = "http://k-db.com/stocks/"
DATASET_PATH = "../dataset/"

# 現在の株式コードのデータを取得
codes = common.get_stockscode()

# 各コードの5年分のデータを取得
for code in codes:
    for year in ["2013","2014","2015","2016","2017"]:
        url_stock = BASE_URL + code + "/1d/" + year + "?download=csv"
        file_path_stock_sjis = DATASET_PATH + "stock_work/" + code + year + ".csv"
        file_path_stock_utf8 = DATASET_PATH + "stock_work/" + code + year + "_utf8.csv"
        file_path_stock_dummy = DATASET_PATH + "stock_work/" + code + year + ".dummy" # データがないときはダミー作成
        if (not((os.path.isfile(file_path_stock_sjis) or os.path.isfile(file_path_stock_dummy)))):
            print(url_stock)
            try:
                urllib.request.urlretrieve(url_stock, file_path_stock_sjis)
                common.sjis_to_utf8(file_path_stock_sjis, file_path_stock_utf8)
            except urllib.error.HTTPError:
                try:
                    time.sleep(3.5)
                    urllib.request.urlretrieve(url_stock, file_path_stock_sjis)
                    common.sjis_to_utf8(file_path_stock_sjis, file_path_stock_utf8)
                except:
                    print("error:", sys.exc_info()[0], " file:", file_path_stock_sjis)
            except:
                print("error:", sys.exc_info()[0], " file:", file_path_stock_sjis)

            time.sleep(3.5)
