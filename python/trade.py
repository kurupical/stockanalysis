# common_library
import numpy as np
import pandas as pd
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
        self.chart = Chart(code, stock_con, date_from, date_to)
        self.decider = \
            Decider(code, tradealgo, predicter)
        self.hold_id = 0
        self.holdstack = pd.DataFrame()
        self.record_id = 0
        self.record = pd.DataFrame()

    def buy(self, amount, limit_price, stop_loss):
        '''
        引数amountだけ株を買う。
        指値limit_price, 逆指値stop_lossを設定する。
        '''
        next = self.chart.get_next()
        # 保有銘柄数の更新
        #DataFrameの初期設定+columnsの順番指定ー＞メモに残したらこのコメント消す
        df = pd.DataFrame({ 'hold_id': self.hold_id,
                            'date': next['日付'],
                            'amount': amount,
                            'price': next['始値'],
                            'limit_price': limit_price,
                            'stop_loss': stop_loss },
                            columns= ['hold_id', 'date', 'amount', 'price', 'limit_price', 'stop_loss'])
        self.holdstack = pd.concat([self.holdstack, df])
        # 取引履歴の更新
        df = pd.DataFrame({ 'record_id': self.record_id,
                            'date': next['日付'],
                            'buysell': "buy",
                            'amount': amount,
                            'price': next['始値']},
                            columns=['record_id','date', 'buysell', 'amount', 'price'])
        self.record = pd.concat([self.record, df])

        self.hold_id += 1
        self.record_id += 1

    def sell(self, amount):
        if self.holdstack['amount'].sum() < amonut:
            print("cannot_sell: don't have stock to sell")
        else:

        print("making now!")

class Decider:
    # 株の予測、意思決定を行う
    def __init__(self, code, tradealgo, predicter):
        self.code = code
        self.tradealgo = tradealgo
        self.predicter = predicter

    def decide_trade(self, predict_term, chart):
        # 売買を決定する
        self.predicter.predict(chart=chart, predict_term=predict_term)
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

    def forward(self):
        '''
        翌営業日のデータを取得し、self.dataに格納
        '''
        next = self.get_next()
        self.data = pd.concat([self.data, next])

    def get_next(self):
        '''
        翌営業日のデータを返す
        '''
        date_today = max(self.df_data['日付'])
        df_after_tomorrow = self.df_all_data[self.df_all_data['日付'] > date_today]
        date_tomorrow = min(df_after_tomorrow['日付'])
        next = df_after_tomorrow[df_after_tomorrow['日付'] == date_tomorrow]
        return next

def test():
    path_ary = [MODEL_PATH + "loss0.089epoch99★UNIT:100-HID:30-lr:0.001-clf:GRU-layer:1.ckpt"]
    predicter = Predicter(path_ary=path_ary)
    tradealgo = [trade_algorithm.UpDown_Npercent(predicter, 10)]
    trade = Trade(code=1301, tradealgo=tradealgo, predicter=predicter, stock_con=stock_con,date_to='2016/12/31')
    judgement = trade.decider.decide_trade(predict_term=30, chart=trade.chart)
    for judge, amount, limit_price, stop_loss in judgement:
        if judge == "buy":
            trade.buy(amount=amount, limit_price=limit_price, stop_loss=stop_loss)
        if judge == "sell":
            trade.sell(amount=amount)

if __name__ == "__main__":
    print("making now!")

    stock_con = learn.StockController()
    stock_con.load()
    test()
    #test
