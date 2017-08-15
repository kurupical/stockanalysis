# common_library
import numpy as np
import configparser
# stockanalysis_library
import learn
import common
import trade_algorithm
MODEL_PATH = "../model/GNUexport/"

class TradeController:
    # 株の保有状況や利益などを管理する。
    def __init__(self):
        print("makng now!")
    def _load_stock(self):
        print("makng now!")

class Trade:
    # チャート予測から株の売買までを行う
    def __init__(self, code, tradealgo, predicter, stock_con, date_from='1900/1/1', date_to='2099/12/31'):
        self.code = code
        self.decider = \
            Decider(code, tradealgo, predicter, stock_con, date_from, date_to)
        self.holdstack = pd.DataFrame(columns=['date', 'amount', 'price', 'limit_price', 'stop_loss'])
        self.record = pd.DataFrame(columns=['date', 'buysell', 'amount', 'price'])

    def trade(self):
        self.decider.decide_trade(predict_term=30)
        print("makng now!")

    def buy():
        print("makng now!")

    def sell():
        print("makng now!")

class Decider:
    # 株の予測、意思決定を行う
    def __init__(self, code, tradealgo, predicter, stock_con, date_from='1900/1/1', date_to='2099/12/31'):
        self.code = code
        self.tradealgo = tradealgo
        self.predicter = predicter
        self.chart = Chart(code, stock_con, date_from, date_to)

    def decide_trade(self, predict_term):
        # 売買を決定する
        self.predicter.predict(chart=self.chart, predict_term=predict_term)
        trade_judge = []
        for algo in self.tradealgo:
            trade_judge.append(algo.judge())
        return trade_judge

class Predicter:
    # 予想機
    def __init__(self, path_ary):
        self.network_ary = []
        for path in path_ary:
            # パラメータファイルを読む
            config = configparser.ConfigParser()
            config.read('test.ini')
            self.unit = int(config['param']['unit'])
            self.n_in = int(config['param']['n_in'])
            self.n_out = int(config['param']['n_out'])
            self.n_hidden = int(config['param']['n_hidden'])
            self.clf = config['param']['clf']
            self.layer = int(config['param']['layer'])
            self.learning_rate = float(config['param']['learning_rate'])
            self.key = config['param']['key']

            network = learn.Network(unit=self.unit,
                                    n_in=self.n_in,
                                    n_out=self.n_out,
                                    n_hidden=self.n_hidden,
                                    clf=self.clf,
                                    layer=self.layer,
                                    learning_rate=self.learning_rate,
                                    key=self.key)
            network.load(path)
            self.network_ary.append(network)

    def predict(self, chart, predict_term=30):
        self.original = chart.df_data['終値'].values
        self.predicted = []
        self.chart = chart
        Z = self.original[:self.unit].reshape(1, self.unit, self.n_in)
        for network in self.network_ary:
            predicted = []
            for i in range(predict_term):
                z_ = Z[-1:]
                y_ = network.y.eval(session=network.sess, feed_dict={
                    network.x: Z[-1:],
                    network.n_batch: 1
                    })

                seq = np.concatenate(
                    (z_.reshape(self.unit, self.n_in)[1:], y_.reshape(1, self.n_in)), axis=0)

                predicted.append(y_.reshape(-1))
                seq = seq.reshape(-1, self.unit, self.n_in)

                Z = np.copy(seq)
            self.predicted.append(predicted)
        print('making now')


class Chart:
    # 株の時系列データを持つ
    def __init__(self, code, stock_con, date_from='1900/1/1', date_to='2099/12/31'):
        self.code = code
        self.stock_obj = stock_con.get_data(code)
        self.df_all_data = self.stock_obj.all_data
        self.df_data = self._get_data(date_from=date_from, date_to=date_to)

    def _get_data(self, date_from='1900/1/1', date_to='2099/12/31'):
        format = '%Y/%m/%d'
        num_from = common.date_to_num(date_from, format)
        num_to = common.date_to_num(date_to, format)
        df = self.df_all_data[self.df_all_data['日付'] >= num_from]
        df = df[df['日付'] <= num_to]
        return df

    def get_next(self):
        # 翌営業日のデータを取得し、self.dataに格納
        date_today = self.df_data.max(columns='日付')
        df_after_tomorrow = self.df_all_data[self.all_data['日付'] > date_today]
        date_tomorrow = df_after_tomorrow.min(columns='日付')
        next = df_after_tomorrow[df_after_tomorrow['日付'] == date_tomorrow]

        self.data = pd.concat([self.data, next])


def test():

    path_ary = [MODEL_PATH + "loss0.033epoch99★UNIT:100-HID:30-lr:0.001-clf:GRU-layer:1.ckpt"]
    predicter = Predicter(path_ary=path_ary)
    tradealgo = [trade_algorithm.UpDown_Npercent(predicter, 10)]
    trade = Trade(code=1301, tradealgo=tradealgo, predicter=predicter, stock_con=stock_con,date_to='2016/12/31')
    trade.trade()

if __name__ == "__main__":
    print("making now!")

    stock_con = learn.StockController()
    stock_con.load()
    test()
    #test
