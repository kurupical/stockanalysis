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
    def __init__(self, init_money):
        # status
        self.money = init_money
        self.holdstock = pd.DataFrame()
        self.history = pd.DataFrame()
        self.id = 0

        self.trade_ary = []
        print("makng now!")

    def add_trade(self,trade_obj):
        self.trade_ary.append(trade_obj)

    def forward_1day(self):
        for trade_obj in self.trade_ary:
            trade_obj.chart.forward_1day()

    def trade(self):
        for trade_obj in self.trade_ary:
            # ---
            # AssetManagementによる売買判定をここで
            # ---
            judgement = trade_obj.decide_trade(predict_term=30)
            # ---
            # amountの調整が必要ならここで
            # ---
            for judge, amount, limit_price, stop_loss in judgement:
                if judge == "buy":
                    df_holdstock, df_history = trade_obj.buy(id=self.id,
                                                            amount=amount,
                                                            limit_price=limit_price,
                                                            stop_loss=stop_loss)
                    self.money -= df_history['price']
                    self.holdstock = pd.concat([self.holdstock, df_holdstock])
                    self.trade_history = pd.concat([self.history, df_history])
                    self.id += 1

                if judge == "sell":
                    df_history = trade_obj.sell(id=self.id, amount=amount)
                    self.trade_history = pd.concat([self.history, df_history])
                    self.money += df_history['price']
                    self.id += 1
                    self._unhold(self.holdstock, code=df_history['code'], amount=df_history['amount'])

    def _unhold(self, code, amount):
        '''
        指定されたコードの指定された株数を減らす
        (先入先出法)
        '''
        while True:
            if amount <= 0:
                break

            min_id = min(self.holdstock[self.holdstock['code'] == code])
            df_min_id = self.holdstock[self.holdstock['id'] == min_id]
            if df_min_id['amount'] <= amount:
                # 該当レコードを削除する
                self.holdstock = self.holdstock[self.holdstock['code' != code]]
                amount -= df_min_id['amount']
            else:
                # 該当レコードの株数を減らす
                df_min_id['amount'] -= amount
                self.holdstock[min_id] = df_min_id
                amount = 0

class Trade:
    # チャート予測から株の売買までを行う
    # →不要な気がする。消すかも
    def __init__(self, code, tradealgo, predicter, stock_con, date_from='1900/1/1', date_to='2099/12/31'):
        self.code = code
        self.chart = Chart(code, stock_con, date_from, date_to)
        self.decider = \
            Decider(code, tradealgo, predicter)

    def decide_trade(self, predict_term):
        return self.decider.decide_trade(predict_term=predict_term, chart=self.chart)

    def buy(self, id, amount, limit_price, stop_loss):
        '''
        引数amountだけ株を買う。
        指値limit_price, 逆指値stop_lossを設定する。
        '''
        next = self.chart.get_next()
        # 保有銘柄数の更新
        #DataFrameの初期設定+columnsの順番指定ー＞メモに残したらこのコメント消す
        df_hold = pd.DataFrame({ 'id': id,
                                 'code': self.code,
                                 'date': next['日付'],
                                 'amount': amount,
                                 'price': next['始値'],
                                 'limit_price': limit_price,
                                 'stop_loss': stop_loss },
                                 columns= ['id', 'code', 'date', 'amount', 'price', 'limit_price', 'stop_loss'])
        # 取引履歴の更新
        df_history = pd.DataFrame({ 'id': id,
                                    'code': self.code,
                                    'date': next['日付'],
                                    'buysell': "buy",
                                    'amount': amount,
                                    'price': next['始値']},
                                    columns=['id', 'code', 'date', 'buysell', 'amount', 'price'])

        return df_hold, df_history

    def sell(self, amount):
        next = self.chart.get_next()
        df_history = pd.DataFrame({ 'id': id,
                                    'code': self.code,
                                    'date': next['日付'],
                                    'buysell': "sell",
                                    'amount': amount,
                                    'price': next['始値']},
                                    columns=['id', 'code', 'date', 'buysell', 'amount', 'price'])
        return df_history

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
            config.read(path + "model.ini")
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
            network.load(path + "model.ckpt")
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

    def forward_1day(self):
        '''
        翌営業日のデータを取得し、self.dataに格納
        '''
        next = self.get_next()
        self.df_data = pd.concat([self.df_data, next])

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
    path_ary = ["../model/1301single/"]
    predicter = Predicter(path_ary=path_ary)
    tradealgo = [trade_algorithm.UpDown_Npercent(predicter, 10)]
    trade = Trade(code=1301, tradealgo=tradealgo, predicter=predicter, stock_con=stock_con,date_to='2016/12/31')
    trade_con = TradeController(1000000)
    trade_con.add_trade(trade)

    trade_con.forward_1day()
    trade_con.trade()
    '''
    judgement = trade.decider.decide_trade(predict_term=30, chart=trade.chart)
    for judge, amount, limit_price, stop_loss in judgement:
        if judge == "buy":
            trade.buy(amount=amount, limit_price=limit_price, stop_loss=stop_loss)
        if judge == "sell":
            trade.sell(amount=amount)
    '''
if __name__ == "__main__":
    print("making now!")

    stock_con = learn.StockController()
    stock_con.load()
    test()
    #test
