import numpy as np
from copy import copy

class UnitRule_Stockcon_Bundle:
    def __init__(self):
        print ("make:", self)
        # 特に処理なし

    def unit(self, stock_obj_ary):
        '''
        学習データ ： stockobj_aryに格納されている銘柄の過去N日分のデータ。
        教師データ : stockobj_aryに格納されている銘柄のうち１つの翌M日分のデータ
        　　　　　　　（全銘柄について学習する。）
        例　学習データ＝１０銘柄、２００日分。
        　　過去５０日分について学習する場合
        　　→データ数は、１０銘柄×（２００−５０）日分
        '''
        stockdata_ary = []
        for stock_obj in stock_obj_ary:
            x, y = stock_obj.unit()
            stockdata_ary.append([x, y])
        '''
        stockdata_ary[][][][]
            １次元目 : 銘柄ごとのインデックス
            2次元目 : [0]=学習データ、[1]=教師データ
            3次元目 : 学習データの単位([１日目〜N日目の株価], [2日目〜N+1日目の株価], ・・・)
            4次元目 : 学習データの内容(1日目の株価、２日目の株価…N日目の株価)
        '''

        is_empty = True
        x = np.array([[[]]])
        y = np.array([[]])
        for i in range(len(stockdata_ary)):
            print("i=:", i)
            target_x, target_y = stockdata_ary[i]

            # target以外の銘柄を取得
            other_x = []
            other_y = []
            for k in range(len(stockdata_ary)):
                print("k:", k)
                if k != i:
                    other_x.append(stockdata_ary[k][0])
                    other_y.append(stockdata_ary[k][1])

            other_x = np.array(other_x).reshape(-1, len(stockdata_ary[0][0][0]), len(stockdata_ary[0][0][0][0]))
            data_x = np.concatenate((target_x, other_x), axis=0)
            data_y = target_y

            # numpy型に変換
            data_x = np.array(data_x).reshape(1, len(data_x), len(data_x[0]), len(data_x[0][0]))
            data_y = np.array(data_y).reshape(len(data_y), len(data_y[0]))

            if is_empty : #配列が空
                x = data_x
                y = data_y
                is_empty = False
            else:
                print("len(x) != 1")
                x = np.concatenate((x, data_x), axis=0)
                y = np.concatenate((y, data_y), axis=0)
        return x, y
