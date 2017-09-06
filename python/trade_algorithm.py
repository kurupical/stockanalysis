# stockanalysis library
from common import *
from logger import *
from configuration import *
from trade import *
# common library
import numpy as np
import math

class Trade_Algorithm:
    @staticmethod
    def generate_trade_algorithm(algo_param, predicter):
        if algo_param[0] == "UpDown_Npercent":
            # param[1] : n_percent
            return Updown_Npercent(predicter=predicter, n_percent=int(algo_param[1]))
        if algo_param[0] == "Updown_Npercent_ClassifyMode":
            # param[1] : n_percent
            return Updown_Npercent_ClassifyMode(predicter=predicter, n_percent=int(algo_param[1]))

class UpDown_Npercent:
    def __init__(self, predicter, n_percent):
        self.predicter = predicter
        self.n_percent = n_percent
        self.logger = Logger(path=Configuration.log_path, obj=self)

    def judge(self):
        '''
        売買判定を行う
        '''

        judge = ""
        amount = 0
        limit_price = 0
        stop_loss = 0
        buy = 0
        sell = 0
        total = len(self.predicter.predicted)

        final_value = self.predicter.chart.stock_obj.stdconv.unstd(self.predicter.original)[-1]
        for predicted in self.predicter.predicted:
            pred_unstd = self.predicter.chart.stock_obj.stdconv.unstd(predicted)
            if final_value * (1 + self.n_percent/100) < np.max(pred_unstd):
                buy += 1
            if final_value * (1 - self.n_percent/100) > np.min(pred_unstd):
                sell += 1


        if buy - sell >= math.floor(total/2):
            judge = "buy"
            amount = 100
            limit_price = int(final_value * (1 + self.n_percent/100))
            stop_loss = int(final_value * (1 - self.n_percent/100))

        if sell - buy >= math.floor(total/2):
            judge = "sell"
            amount = 100
            limit_price = 0
            stop_loss = 0


        # ログ出力
        date_num = self.predicter.chart.get_today_date()
        date = num_to_date(num=self.predicter.chart.get_today_date(), format="%Y/%m/%d")
        log = "code:" + str(self.predicter.chart.code) + \
              ",date:" + str(date) + \
              ",end_val:" + str(final_value) + \
              ",predict_max:" + str(np.max(pred_unstd)) + \
              ",predict_min:" + str(np.min(pred_unstd)) + \
              ",judge:" + judge + \
              ",amount:" + str(amount) + \
              ",limit_price:" + str(limit_price) + \
              ",stop_loss:" + str(stop_loss)
        self.logger.log(log)

        return judge, amount, limit_price, stop_loss

class Updown_Npercent_ClassifyMode:
    def __init__(self, predicter, n_percent):
        self.predicter = predicter
        self.logger = Logger(path=Configuration.log_path, obj=self)
        self.n_percent = n_percent

    def judge(self):
        judge = ""
        amount = 0
        limit_price = 0
        stop_loss = 0
        buy = 0
        sell = 0

        final_value = self.predicter.chart.stock_obj.stdconv.unstd(self.predicter.original)[-1]
        # self.predicter.predictedは予想[sell, stay, buy]の確率。　最大値のidxを取得
        judge = np.argmax(self.predicter.predicted.reshape(-1), axis=0)

        if judge == 0: # sell
            judge = "sell"
            amount = 100
            limit_price = 0
            stop_loss = 0

        if judge == 1: # stay
            judge = ""

        if judge == 2: # buy
            judge = "buy"
            amount = 100
            limit_price = int(final_value * (1 + self.n_percent/100))
            stop_loss = int(final_value * (1 - self.n_percent/100))

        # ログ出力
        date_num = self.predicter.chart.get_today_date()
        date = num_to_date(num=self.predicter.chart.get_today_date(), format="%Y/%m/%d")
        log = "code:" + str(self.predicter.chart.code) + \
              ",date:" + str(date) + \
              ",end_val:" + str(final_value) + \
              ",judge:" + judge + \
              ",amount:" + str(amount) + \
              ",limit_price:" + str(limit_price) + \
              ",stop_loss:" + str(stop_loss)
        self.logger.log(log)

        return judge, amount, limit_price, stop_loss
