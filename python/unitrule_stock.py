# stockanalysis library
from common import *

# common library
import numpy as np
import pandas as pd

class UnitRule_Stock_ForwardDay:
    def __init__(self, unit_amount, forward_day, predict_mode, classify_ratio=None):
        '''
        株データを単位ごとに分割するルールを定義
        unit_amount  : １単位あたりのデータ数
        forward_day  : 予想対象の日にち（学習データから過ぎた日数）
        predict_mode : 予想する対象
            ・nornal
             forward_dayで指定した日数後の、output_itemsで指定しているデータを予想
            ・max_min
             学習データの最終日からforward_dayで指定した日数間の中で、output_itemsで指定したデータ
             最大値・最小値を予想（output_items="終値"の時のみ動作保証）
        '''
        self.unit_amount = unit_amount
        self.forward_day = forward_day
        self.predict_mode = predict_mode
        self.classify_ratio = classify_ratio

    def unit(self, stock_obj):
        x = np.array([[[]]])
        y = np.array([[]])

        data = []
        target = []

        data_ary = stock_obj.data.values
        df_tag = stock_obj.all_data

        if len(data_ary) > self.unit_amount:
            for i in range(0, len(data_ary) - self.unit_amount - self.forward_day):
                data.append(data_ary[i:i + self.unit_amount, :])
                if self.predict_mode == "normal":
                    target.append(data_ary[i + self.unit_amount + self.forward_day - 1, :len(stock_obj.output_items)])
                if self.predict_mode == "max_min":
                    pred_ary = data_ary[i + self.unit_amount : i + self.unit_amount + self.forward_day, :len(stock_obj.output_items)]
                    max_value = np.max(pred_ary)
                    min_value = np.min(pred_ary)
                    target.append([max_value, min_value])
                if self.predict_mode == "max_min_classify":
                    pred_ary = data_ary[i + self.unit_amount : i + self.unit_amount + self.forward_day, :len(stock_obj.output_items)]
                    max_value = stock_obj.stdconv.unstd(np.max(pred_ary))
                    min_value = stock_obj.stdconv.unstd(np.min(pred_ary))
                    final_value = stock_obj.stdconv.unstd(data[i][-1])
                    max_value_ratio = (max_value - final_value) / final_value
                    min_value_ratio = (min_value - final_value) / final_value
                    w_target = []
                    w_target.append(self._classify(max_value_ratio))
                    #w_target.append(self._classify(min_value_ratio))
                    target.append(np.array(w_target).reshape(-1))
                tag = self._tag(df=df_tag.ix[i,:], stock_obj=stock_obj, unit_amount=self.unit_amount, idx=i)
            if len(x) == 1:
                x = np.array(data).reshape(len(data), self.unit_amount, len(data[0][0]))
                y = np.array(target).reshape(len(target),len(target[0]))
                ret_tag = tag
            else:
                x = np.concatenate((x, np.array(data).reshape(len(data), self.unit_amount, len(data[0][0]))), axis=0)
                y = np.concatenate((y, np.array(target).reshape(len(data), len(target[0]))), axis=0)
                ret_tag = pd.concat(ret_tag, tag)

        return x, y, ret_tag

    def _tag(self, df, unit_amount, stock_obj, idx):
        '''
        データのタグを返す
        '''
        # start_date
        start_date = df.loc["日付"]
        start_date = num_to_date(num=int(start_date), format="%Y/%m/%d")

        # start_idx
        start_idx = idx

        # 銘柄
        code = int(stock_obj.code)

        df = pd.DataFrame({ 'start_date': [start_date],
                            'start_idx': [start_idx],
                            'code': [code],
                            'unit_amount': [unit_amount]})
        return df

    def _classify(self, ratio):
        '''
        データを分類
        '''

        if ratio <= -1 * self.classify_ratio:
            return 1,0,0
        if ratio < self.classify_ratio and ratio > -1 * self.classify_ratio:
            return 0,1,0
        if ratio >= self.classify_ratio:
            return 0,0,1
