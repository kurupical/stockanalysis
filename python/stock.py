import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from common import *
from configparser import *

class Stock:
    def __init__(self,
                read_data=None,
                isStdmode=True,
                isUpdownratiomode=False,
                input_items=None,
                output_items=None,
                unitrule_stock=None):
        self.isStdmode = isStdmode
        self.isUpdownratiomode = isUpdownratiomode
        self.input_items = input_items
        self.output_items = output_items
        self.unitrule_stock = unitrule_stock
        try:
            codes = read_data["証券コード"].values
            self.code = codes[0]
        except KeyError: # 株以外のデータをテストデータとして使用するとき
            print("KeyError: DEBUG_MODE")
            self.input_items=['0']
            self.output_items=['0']
            self.code = ANALYSIS_CODE
        self.all_data = read_data
        data = read_data[self.input_items]

        # 標準化データ(平均=0,標準偏差=1)
        self.data = data

        if isStdmode:
            self.stdconv = StdConverter(self.data)
            self.data = self.stdconv.data_std

        if isUpdownratiomode:
            for str in self.input_items:
                w_ary = np.copy(data[str].values)
                ary = []
                for i in range(len(w_ary)):
                    if i == 0:
                        continue
                    if i == 1:
                        ary.append(1)
                        ary.append(w_ary[i-1] / w_ary[i])
                    if i > 1:
                        ary.append(w_ary[i-1] / w_ary[i])
                self.data[str] = ary

        for str in self.input_items:
            self.all_data[str] = self.data[str]

    def unit(self):
        ary = self.data.values
        self.x, self.y, self.tag = self.unitrule_stock.unit(self)

    # def convertupdownratio(self, ary):

    def get_index(self, item_name):
        if len(item_name) == 1:
            return self.input_items.index(item_name[0])
        else:
            # 未実装
            return None


class StockController:
    def __init__(self,
                 csv_path,
                 unitrule_stock,
                 unitrule_stockcon,
                 stock_info,
                 input_items,
                 output_items):
        self.csv_path = csv_path
        self.unitrule_stock = unitrule_stock
        self.unitrule_stockcon = unitrule_stockcon
        self.stock_info = stock_info
        self.input_items = input_items
        self.output_items = output_items
        self.stockdata = [] # Stockオブジェクトを格納するlist

    def load(self):
        input_path = self.csv_path + "*.csv"
        files = glob.glob(input_path)
        print ("load")
        pbar = tqdm(total=len(files))
        data = np.array([[]])
        for file in files:
            read_data = pd.read_csv(file)
            if (len(read_data.index) != 0):
                stock = Stock(read_data=read_data,
                              input_items=self.input_items,
                              output_items=self.output_items,
                              unitrule_stock=self.unitrule_stock)
                self.stockdata.append(stock)
            pbar.update(1)
        pbar.close()

    def search_high_cor(self, cor, code, unit):
        '''
         指定した銘柄の最新からunit日前〜0日前のデータと相関の高い
         「最新からunit*2日前〜unit+1日前の銘柄」をstockdataにセットする
         (指定されたコードもセットする)
         param
            cor : 基準となる相関係数
            code : 指定する銘柄
            unit : １銘柄あたりのデータの単位(unit日分)
        '''
        ary = []
        x = self.get_data(code)
        x = x.data[-unit:]
        print ("search_high_cor")
        amount_of_search = len(self.stockdata)
        pbar = tqdm(total=len(self.stockdata))
        for stock_obj in self.stockdata:
            if stock_obj.code == code:
                stock_obj.data = stock_obj.data[-unit*2:]
                ary.append(stock_obj)
            if len(stock_obj.data) > unit*2:
                y = stock_obj.data[-unit*2:-unit]
                xy_cor = np.corrcoef(x.values.reshape(-1), y.values.reshape(-1))[0][1]
                if abs(xy_cor) > abs(cor):
                    stock_obj.data = stock_obj.data[-unit*2:]
                    ary.append(stock_obj)
                    print("証券コード:", stock_obj.code, " 相関係数:", xy_cor)
            pbar.update(1)
        pbar.close()
        self.stockdata = ary
        print("search_high_cor 結果:\n")
        print("*******************************************")
        print("分析銘柄数:", amount_of_search)
        print("高相関銘柄:", len(self.stockdata))
        print("*******************************************")

    def search_isinrange_marketcap(self, min_value=0, max_value=10*20):
        '''
        時価総額(marketcap)がmin_valueとmax_valueの間にある銘柄のみを抽出する。
        '''
        ary = []
        amount_of_search = len(self.stockdata)

        for stock_obj in self.stockdata:
            df = self.stock_info.get_info(code=stock_obj.code)
            # 発行済株数
            if df["一株当り純利益"].values == 0:
                print("error! 一株あたり純利益データなし: code=", str(stock_obj.code))
            elif len(df.index) == 0:
                print("error! 銘柄データなし: code=", str(stock_obj.code))
            elif len(df.index) > 1:
                print("error! 銘柄データ複数: code=", str(stock_obj.code))
            else:
                #　本来はここでreplaceではない。。（csvがすでに綺麗な形になっているのがベスト）
                issued = float(max(df["純利益"].values).replace(",", "").replace(" ", "")) \
                        / float(max(df["一株当り純利益"].values).replace(",", ""))
                # 末尾の株価
                value = stock_obj.stdconv.unstd(stock_obj.all_data["終値"].values)
                value = float(value[-1])
                if (issued * value > min_value and issued * value < max_value):
                    ary.append(stock_obj)
                    print("capOK! code:" + str(stock_obj.code) + " value:" + str(value), " issued:" + str(issued))

        self.stockdata = ary

        print("search_isinrange_marketcap 結果:\n")
        print("*******************************************")
        print("分析銘柄数:", amount_of_search)
        print("抽出銘柄数:", len(self.stockdata))
        print("*******************************************")

    def search_is_YMDbefore(self, ymd):
        ary = []
        amount_of_search = len(self.stockdata)

        for stock_obj in self.stockdata:
            date_num= date_to_num(ymd, "%Y/%m/%d")
            stock_obj.all_data = stock_obj.all_data[stock_obj.all_data["日付"] <= date_num]
            stock_obj.data = pd.DataFrame(stock_obj.all_data["終値"])

            ary.append(stock_obj)

        self.stockdata = ary
        print("search_is_YMDbefore 結果:\n")
        print("*******************************************")
        print("分析銘柄数:", amount_of_search)
        print("抽出銘柄数:", len(self.stockdata))
        print("*******************************************")

    def search_isexist_past_Nday(self, n_day):
        '''
        過去N日分のデータが存在する株のみを抽出し、self.stockdataに格納する。
        ※歯抜けデータは考慮していません・・・
        '''
        ary = []
        amount_of_search = len(self.stockdata)

        for stock_obj in self.stockdata:
            if len(stock_obj.data) > n_day:
                stock_obj.data = stock_obj.data[-n_day:]
                ary.append(stock_obj)

        self.stockdata = ary
        print("isexist_past_Nday 結果:\n")
        print("*******************************************")
        print("分析銘柄数:", amount_of_search)
        print("抽出銘柄数:", len(self.stockdata))
        print("*******************************************")


    def unit_data(self):
        self.data_x, self.data_y, self.tag = self.unitrule_stockcon.unit(self.stockdata)

        print("unit_data 結果:\n")
        print("*******************************************")
        print("分析銘柄数:", len(self.stockdata))
        print("入力データ:", len(self.data_x))
        print("*******************************************")

    def get_data(self, code):
        for i in range(len(self.stockdata)):
            stock_obj = self.stockdata[i]
            if stock_obj.code == code:
                return stock_obj

    def get_stockcode_ary(self):
        '''
        StockControllerが持っているStockオブジェクトのコードを返す
        '''
        code_ary = []
        for stock_obj in self.stockdata:
            code_ary.append(stock_obj.code)
        return code_ary

    def save_config(self, path):
        config = ConfigParser()
        # input_itemsをカンマ区切りで編集
        input_items = ""
        for item in self.input_items:
            input_items = input_items + item + ","
        # output_itemsをカンマ区切りで編集
        output_items = ""
        for item in self.output_items:
            output_items = output_items + item + ","
        config['param'] = { 'csv_path':          self.csv_path,
                            'unitrule_stock':    self.unitrule_stock.__class__.__name__,
                            'unitrule_stockcon': self.unitrule_stockcon.__class__.__name__,
                            'stock_info_path':   self.stock_info.path,
                            'input_items':       input_items,
                            'output_items':      output_items}
        with open(path + 'stock_con.ini', 'w') as configfile:
            config.write(configfile)


class StockInfo:
    def __init__(self, path):
        self.path = path
        self._load_info()

    def _load_info(self):
        self.df_data = pd.read_csv(self.path)

    def get_info(self, code, mode="recent"):
        df = self.df_data[self.df_data["証券コード"] == code]
        max_date = max(df["期末"])
        if mode == "recent":
            df = df[df["期末"] == max_date]
            if len(df.index) > 1 :
                df = df.loc[(df['連結個別'] == '連結')]

        return df
