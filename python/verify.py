import math
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from learn import StockController
from learn import Stock
import random
import pandas as pd
import csv

SIZE = 300  # データサイズ
SEQUENCE_LENGTH = 100

def run():
    def divive_sequence(x):
        y = []
        for i in range(SIZE - SEQUENCE_LENGTH + 1):
            y.append(x[i:i+SEQUENCE_LENGTH])
        return y

    def calc_cor(x):
        cor = np.corrcoef(x)
        return cor

    def calc_cor_ave(x):
        count = 0
        cor_total = 0
        for i in range(len(x)):
            for k in range(len(x)):
                if i != k:
                    count += 1
                    cor_total += x[i,k]

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

    def random_ndim_func(step=10000, dim=None):
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
if __name__ == '__main__':
    run()
