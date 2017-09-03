# stockanalysis library
from predicter import *
from trade import *
from configuration import *
# common library
from random import *
from copy import *
import matplotlib.pyplot as plt

class VerifyModel:
    '''
    learn.pyで学習したモデルを検証するためのクラス
    '''
    @staticmethod
    def generate_verify_model(network, stock_con):
        config = Configuration.parse_from_file()
        verify_model = config['param']['verify_model']

        if verify_model == "VerifyModel_MaxMin_Graph":
            return VerifyModel_MaxMin_Graph(network, stock_con, config)
        if verify_model == "VerifyModel_MaxMin_Classify":
            return VerifyModel_MaxMin_Classify(network, stock_con, config)

    def __init__(self, network, stock_con):
        self.network = network
        self.stock_con = stock_con

    def get_random_datefrom(self):
        stock_obj = self.stock_con.stockdata[0]

        w_df = stock_obj.all_data["日付"].head(len(stock_obj.x))
        choice_date = np.random.choice(w_df.values, 1)

        # 数値型で返す
        return choice_date[0]

    def _generate_charts(self, date_from):
        # chartの作成
        charts = []
        date_format="%Y/%m/%d"
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

        return charts

class VerifyModel_MaxMin_Graph(VerifyModel):
    def __init__(self, network, stock_con, config):
        super().__init__(network, stock_con)
        self.times = int(config['param']['verify_times'])

    def verify(self, path, date_from=None):
        if date_from is None:
            date_from = super().get_random_datefrom()
        # 数値で来た場合は日付に変換
        if date_from.__class__.__name__ != "str":
            date_from = num_to_date(num=date_from, format=date_format)
        charts = super()._generate_charts(date_from)

        for i in range(self.times):
            # predicterの作成
            if self.stock_con.unitrule_stockcon.__class__.__name__ == "UnitRule_Stockcon_Bundle":
                model = "Predicter_Nto1Predict_MaxMin"
            if self.stock_con.unitrule_stockcon.__class__.__name__ == "UnitRule_Stockcon_Normal":
                model = "Predicter_1to1Predict_MaxMin"
            predicter = Predicter.generate_predicter(model=model,
                                                     network_ary=[self.network])

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

class VerifyModel_MaxMin_Classify(VerifyModel):
    def __init__(self, network, stock_con, config):
        super().__init__(network, stock_con)

    def verify(self, path):
        # predicterの作成
        if self.stock_con.unitrule_stockcon.__class__.__name__ == "UnitRule_Stockcon_Bundle":
            model = "Predicter_Nto1Predict_MaxMin"
        if self.stock_con.unitrule_stockcon.__class__.__name__ == "UnitRule_Stockcon_Normal":
            model = "Predicter_1to1Predict_MaxMin"

        total_pred_result_table = np.zeros((3,3))
        for stock_obj in self.stock_con.stockdata:
            pred_result_table = np.zeros((3,3))
            for x, y in zip(stock_obj.x, stock_obj.y):
                Z = x.reshape(1, self.network.unit_amount, self.network.n_in)
                y_ = self.network.y.eval(session=self.network.sess, feed_dict={
                    self.network.x: Z[-1:],
                    self.network.n_batch: 1
                })

                # print("y_:" , y_, "y:", y)
                y_pred = np.argmax(y_.reshape(-1), axis=0)
                y_result = np.argmax(y, axis=0)

                pred_result_table[y_pred, y_result] += 1
            print("code:", stock_obj.code)
            print(pred_result_table)
            total_pred_result_table += pred_result_table
        print("縦軸：予想, 横軸:実際の結果")
        print("total:\n", total_pred_result_table)
