# stockanalysis library
from predicter import *
from trade import *
# common library
from random import *
from copy import *
import matplotlib.pyplot as plt

class VerifyModel:
    '''
    learn.pyで学習したモデルを検証するためのクラス
    '''
    def __init__(self, network, stock_con):
        self.network = network
        self.stock_con = stock_con

    def get_random_datefrom(self):
        stock_obj = self.stock_con.stockdata[0]

        w_df = stock_obj.all_data["日付"].head(len(stock_obj.x))
        choice_date = np.random.choice(w_df.values, 1)

        # 数値型で返す
        return choice_date[0]

    def maxmin_graph_verify(self, times, date_from, path):

        date_format="%Y/%m/%d"

        # 数値で来た場合は日付に変換
        if date_from.__class__.__name__ != "str":
            date_from = num_to_date(num=date_from, format=date_format)

        for i in range(times):
            # predicterの作成
            if self.stock_con.unitrule_stockcon.__class__.__name__ == "UnitRule_Stockcon_Bundle":
                model = "Predicter_Nto1Predict_MaxMin"
            if self.stock_con.unitrule_stockcon.__class__.__name__ == "UnitRule_Stockcon_Normal":
                model = "Predicter_1to1Predict_MaxMin"
            predicter = Predicter.generate_predicter(model=model,
                                                     network_ary=[self.network])
            # chartの作成
            charts = []
            for stock_obj in self.stock_con.stockdata:
                # from_dateを起点とし指定した日数分のチャートを生成
                w_df = stock_obj.all_data[stock_obj.all_data["日付"] >= date_to_num(date=date_from, format=date_format)]
                unit_amount = stock_obj.unitrule_stock.unit_amount
                date_to = w_df["日付"].values[unit_amount]
                date_to = num_to_date(num=date_to, format=date_format)

                chart = Chart(stock_obj=stock_obj,
                              date_from=date_from,
                              date_to=date_to)
                charts.append(chart)

            # 各銘柄の予実を表示
            for stock_obj in self.stock_con.stockdata:
                # 予想
                predicted = predicter.predict(charts=charts,
                                              code=stock_obj.code)

                # 実績(forward_day分だけ進める)
                forward_day = stock_obj.unitrule_stock.forward_day
                ary = []
                chart = Chart.get_chart(charts=charts, code=stock_obj.code)
                for i in range(forward_day):
                    chart.forward_1day()

                # 表示(グラフ)
                data = chart.get_value_data()
                max_min = stock_obj.stdconv.unstd(predicter.predicted)
                max_value = np.max(max_min)
                min_value = np.min(max_min)
                plt.figure()
                plt.plot(data["終値"].values, color='black', label="end_value")
                plt.legend()
                plt.xlabel('date')
                plt.ylabel('value')
                x_max = len(chart.get_value_data())
                x_min = x_max - forward_day
                plt.hlines([min_value, max_value], x_min, x_max, linestyles="dashed")
                filename = path + str(stock_obj.code) + ".jpeg"
                title = "Code:" + str(stock_obj.code) + "  start_date:" + str(np.min(data["日付"].values))
                plt.title(title)
                plt.savefig(filename)
                print("code:", stock_obj.code)
