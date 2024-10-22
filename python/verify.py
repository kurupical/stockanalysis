import math
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from learn import StockController
from learn import Stock
import random
import pandas as pd
import csv
import common
import glob
import shutil

SIZE = 1000  # データサイズ
SEQUENCE_LENGTH = 100

def divive_sequence(x):
    y = []
    for i in range(SIZE - SEQUENCE_LENGTH + 1):
        y.append(x[i:i+SEQUENCE_LENGTH])
    return y

def calc_cor(x):
    cor = np.corrcoef(x)
    cor = np.round(cor, 3)
    return cor

def calc_cor_ave(x, abs_mode=True):
    count = 0
    cor_total = 0
    for i in range(len(x)):
        for k in range(len(x)):
            if i != k and not math.isnan(x[i,k]):
                count += 1
                if abs_mode:
                    cor = abs(x[i,k])
                else:
                    cor = x[i,k]
                cor_total += cor

    coef_ave = cor_total / count
    return coef_ave

def root_x_func(start=0):
    x = []
    for i in range(SIZE):
        x.append(math.sqrt(i + start))

    y = divive_sequence(x)
    cor = calc_cor(y)
    cor_ave = calc_cor_ave(cor)

    # テストデータ出力
    df = pd.DataFrame(x)
    df.to_csv('test_rootx-COR=' + str(round(cor_ave,3)) + '.csv')
    #sns.heatmap(cor, annot=False, xticklabels='',yticklabels='')
    #plt.title("y=√x     seq_length=" + str(SEQUENCE_LENGTH) + "  data_size=" + str(SIZE))
    #plt.show()
    print ("root_x_func cor_ave=" + str(round(cor_ave,3)))
    return ["root_x_func", round(cor_ave,3)]

def random_ndim_func(dim=None, step=2, name="nfunc_data.csv"):
    def make_func(a=None,b=None,c=None,d=None,e=None,dim=dim):
        if a == None:
            a = random.uniform(SIZE*(-1/100), SIZE/100)
            b = random.uniform(SIZE*(-1/100), SIZE/100)
            c = random.uniform(SIZE*(-1/100), SIZE/100)
            d = random.uniform(SIZE*(-1/100), SIZE/100)
            e = random.uniform(SIZE*(-1/100), SIZE/100)
        if dim == 2:
            return lambda x: a*x**2 + b*x + c
        if dim == 3:
            return lambda x: a*x**3 + b*x**2 + c*x + d
        if dim == 4:
            return lambda x: a*x**4 + b*x**3 + c*x**2 + d*x + e

    f = make_func()
    x = np.arange(SIZE/(-200), SIZE/200, 0.01)
    y = f(x)

    # テストデータ出力
    df = pd.DataFrame(y)
    df.to_csv(name, index=False)
    cor_ave_total = 0
    for i in range(step):
        f = make_func()
        x = np.arange(SIZE/(-200), SIZE/200, 0.01)
        y = f(x)
        div_y = divive_sequence(y)
        cor = calc_cor(div_y)
        cor_ave_total += calc_cor_ave(cor)

    # テストデータ出力
    df = pd.DataFrame(y)
    df.to_csv('test_' + str(dim) + 'dim-COR=' + str(round(calc_cor_ave(cor),3)) + '.csv')

    cor_ave = cor_ave_total / step
    print ("dim=" + str(dim))
    print ("cor_ave=" + str(round(cor_ave,3)))
    return ["dim=" + str(dim) + "func", round(cor_ave,3)]

def stock_func():
    ary = []
    stock_con = StockController()
    stock_con.load()
    for stock_obj in stock_con.stockdata:
        if len(stock_obj.data) > SIZE:
            x, y = stock_obj.unit(SEQUENCE_LENGTH)
            cor = calc_cor(x.reshape(-1, SEQUENCE_LENGTH))
            cor_ave = calc_cor_ave(cor)
            #sns.heatmap(cor, annot=False, xticklabels='',yticklabels='')
            #plt.show()
            print("stock code=" + str(stock_obj.code) + " cor_ave=" + str(round(cor_ave,3)))
            ary.append([stock_obj.code, round(cor_ave,3)])
    print("stock_cor_avg:" , str(len(stock_con.stockdata)))
    return ary

def func_to_csv():
    cor_ary = []
    cor_ary.append(root_x_func())
    w_ary = stock_func()
    for i in range(len(w_ary)):
        cor_ary.append(w_ary[i])
    cor_ary.append(random_ndim_func(dim=2))
    cor_ary.append(random_ndim_func(dim=3))
    cor_ary.append(random_ndim_func(dim=4))
    for i in range(len(cor_ary)):
        print(cor_ary[i])

    df = pd.DataFrame(cor_ary)
    df.to_csv('result.csv')

def verify_cor(cor=None, code=None, unit=100):
    stock_con = StockController()
    stock_con.load()
    # 検索したい条件を記述
    if cor != None:
        stock_con.search_high_cor(cor=cor, code=code, unit=unit)

    ary = []
    for stock_obj in stock_con.stockdata:
        if not stock_obj.code == code:
            ary.append(stock_obj.data.values[-unit:].reshape(-1))

    ary = np.array(ary)
    cor_ary = calc_cor(ary)
    cor_ave = calc_cor_ave(cor_ary)

    print("\n\n\n\n\n\n\n\n")
    print("***************************************")
    print("code=", str(code))
    print("condition: cor>", str(cor))
    print("cor:")
    print(cor_ary)
    print("cor_ave:", str(cor_ave))
    print("***************************************")

def check_invalid_data():
    csv_path = "../dataset/stock_analysis/"
    stockinfo_path = "../dataset/stock_info.csv"
    input_items = ["終値"]
    output_items = ["終値"]

    unitrule_s = 1
    unitrule_sc = 1
    stock_info = 1
    stock_con = StockController(csv_path=csv_path,
                                unitrule_stock=unitrule_s,
                                unitrule_stockcon=unitrule_sc,
                                stock_info=stock_info,
                                input_items=input_items,
                                output_items=output_items,
                                isStdmode=False)
    stock_con.load()

    for stock_obj in stock_con.stockdata:
        msg1 = "OK"
        msg2 = "OK"
        stock_1daybefore = 0
        for key, stock in stock_obj.data.iterrows():
            if stock_1daybefore != 0:
                # 前日比２倍以上
                if stock_1daybefore * 2 < stock['終値'] and msg1 == "OK":
                    print("＜前日比２倍以上＞銘柄:", stock_obj.code)
                    msg1 = "NG"

                # 前日比1/2以下
                if stock_1daybefore / 2 > stock['終値']*1.02 and msg1 == "OK":
                    print("＜前日比半分以下＞銘柄:", stock_obj.code)
                    msg1 = "NG"
            stock_1daybefore = stock['終値']

            # 金額が異常値
            # if stock_obj.stdconv.unstd(np.array([stock['終値']])) > 10**8:
            if stock['終値'] > 10**8 and msg2 == "OK":
                print("＜金額異常＞銘柄:", stock_obj.code)
                msg2 = "NG"

            if msg1 == "NG" or msg2=="NG":
                input_path = "../dataset/stock_analysis/" + str(stock_obj.code) + "*.csv"
                files = glob.glob(input_path)
                for file in files:
                    shutil.move(file, "../dataset/stock_analysis/NGData/")


if __name__ == '__main__':
    # 動かしたいメソッドを記述
    # verify_cor(cor=0.6, code=1301)
    # check_invalid_data()
    random_ndim_func(dim=3)
    random_ndim_func(dim=4)
