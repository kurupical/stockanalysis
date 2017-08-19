

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
            x, y = stock_obj.unit(stock_obj)
            stockdata_ary.append([stock_obj.code, x, y])
        print(stockdata_ary)
