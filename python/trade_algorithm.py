# stockanalysis library
from common import *
from logger import *
from configuration import *
# common library
import numpy as np
import math

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

        org_unstd = self.predicter.chart.stock_obj.stdconv.unstd(self.predicter.original)
        for predicted in self.predicter.predicted:
            pred_unstd = self.predicter.chart.stock_obj.stdconv.unstd(predicted)
            if org_unstd[-1] * (1 + self.n_percent/100) < np.max(pred_unstd):
                buy += 1
            if org_unstd[-1] * (1 - self.n_percent/100) > np.min(pred_unstd):
                sell += 1


        if buy - sell >= math.floor(total/2):
            judge = "buy"
            amount = 100
            limit_price = int(org_unstd[-1] * (1 + self.n_percent/100))
            stop_loss = int(org_unstd[-1] * (1 - self.n_percent/100))

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
              ",end_val:" + str(org_unstd[-1]) + \
              ",predict_max:" + str(np.max(pred_unstd)) + \
              ",predict_min:" + str(np.min(pred_unstd)) + \
              ",judge:" + judge + \
              ",amount:" + str(amount) + \
              ",limit_price:" + str(limit_price) + \
              ",stop_loss:" + str(stop_loss)
        self.logger.log(log)

        return judge, amount, limit_price, stop_loss
