# stockanalysis library
import trade
import learn
# common library
import numpy as np
import math

class UpDown_Npercent:
    def __init__(self, predicter, n_percent):
        self.predicter = predicter
        self.n_percent = n_percent

    def judge(self):
        '''
        売買判定を行う
        '''

        judge = ""
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

        if buy - sell >= math.floor(total):
            judge = "buy"

        if sell - buy >= math.floor(total):
            judge = "sell"
        return judge
