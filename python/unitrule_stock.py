import numpy as np

class UnitRule_Stock_ForwardDay:
    def __init__(self, unit_amount, forward_day, predict_mode):
        '''
        株データを単位ごとに分割するルールを定義
        unit_amount  : １単位あたりのデータ数
        forward_day  : 予想対象の日にち（学習データから過ぎた日数）
        predict_mode : 予想する対象
            ・nornal
             forward_dayで指定した日数後の、output_itemsで指定しているデータを予想
            ・max_min
             学習データの最終日からforward_dayで指定した日数間の中で、output_itemsで指定したデータの
             最大値・最小値を予想（output_items="終値"の時のみ動作保証）
        '''
        self.unit_amount = unit_amount
        self.forward_day = forward_day
        self.predict_mode = predict_mode

    def unit(self, stock_obj):
        x = np.array([[[]]])
        y = np.array([[]])

        data = []
        target = []
        ary = stock_obj.data.values

        if len(ary) > self.unit_amount:
            for i in range(0, len(ary) - self.unit_amount - self.forward_day):
                data.append(ary[i:i + self.unit_amount, :])
                if self.predict_mode == "normal":
                    target.append(ary[i + self.unit_amount + self.forward_day - 1, :len(stock_obj.output_items)])
                if self.predict_mode == "max_min":
                    pred_ary = ary[i + self.unit_amount, :self.forward_day, :len(stock_obj.output_items)]
                    max_value = np.max(pred_ary)
                    min_value = np.min(pred_ary)
                    target = [max_value, min_value]
            if len(x) == 1:
                x = np.array(data).reshape(len(data), self.unit_amount, len(data[0][0]))
                y = np.array(target).reshape(len(target),len(target[0]))
            else:
                x = np.concatenate((x, np.array(data).reshape(len(data), self.unit_amount, len(data[0][0]))), axis=0)
                y = np.concatenate((y, np.array(target).reshape(len(data), len(target[0]))), axis=0)

        return x, y