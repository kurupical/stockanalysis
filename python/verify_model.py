# stockanalysis library
from predicter import *
from trade import *
from configuration import *
from logger import *
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
        verify_model = Configuration.config['param']['verify_model']

        if verify_model == "VerifyModel_MaxMin_Graph":
            return VerifyModel_MaxMin_Graph(network, stock_con)
        if verify_model == "VerifyModel_MaxMin_Classify":
            return VerifyModel_MaxMin_Classify(network, stock_con)

    def __init__(self, network, stock_con):
        self.network = network
        self.stock_con = stock_con
        self.logger = Logger(Configuration.log_path)

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
    def __init__(self, network, stock_con):
        super().__init__(network, stock_con)
        self.times = int(Configuration.config['param']['verify_times'])

    def verify(self, path, date_from=None):
        date_format="%Y/%m/%d"
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
    def __init__(self, network, stock_con):
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
            self.logger.log("code:" + str(stock_obj.code))
            self.logger.log("\n%s" % pred_result_table)
            total_pred_result_table += pred_result_table

        output_log = "total:\n%s" % (total_pred_result_table)
        self.logger.log("(予想/結果)")
        self.logger.log("(Buy:N%以上up/Sell:N%以上down/Stay:それ以外)")
        self.logger.log("(Sell/Sell):" + str(total_pred_result_table[0,0]))
        self.logger.log("(Sell/Stay):" + str(total_pred_result_table[0,1]))
        self.logger.log("(Sell/Buy):" + str(total_pred_result_table[0,2]))
        self.logger.log("(Stay/Sell):" + str(total_pred_result_table[1,0]))
        self.logger.log("(Stay/Stay):" + str(total_pred_result_table[1,1]))
        self.logger.log("(Stay/Buy):" + str(total_pred_result_table[1,2]))
        self.logger.log("(Buy/Sell):" + str(total_pred_result_table[2,0]))
        self.logger.log("(Buy/Stay):" + str(total_pred_result_table[2,1]))
        self.logger.log("(Buy/Buy):" + str(total_pred_result_table[2,2]))
        correct_buy = total_pred_result_table[2,2] / \
                      (total_pred_result_table[0,2] + total_pred_result_table[1,2] + total_pred_result_table[2,2])
        wrong_buy = (total_pred_result_table[2,0] + total_pred_result_table[2,1]) / \
                    (total_pred_result_table[2,0] + total_pred_result_table[2,1] + total_pred_result_table[2,2])
        self.logger.log("「買い」タイミングで正しく買えた率:" + str(correct_buy))
        self.logger.log("「買い」でないタイミングで誤って買った率:" + str(wrong_buy))
        print("縦軸：予想, 横軸:実際の結果")
        print(output_log)
