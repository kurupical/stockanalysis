from random import *

class VerifyModel:
    '''
    learn.pyで学習したモデルを検証するためのクラス
    '''
    def __init__(self, network, stock_con):
        self.network = network
        self.stock_con = stock_con

    def maxmin_graph_verify(self, times):
        for i in range(times):
            idx = randint(0, len(stock_con.stockdata[0].data_x)
            for stock_obj in stock_con.stockdata:
                # テスト項目の設定
                x = stock_obj.data_x[idx]

            # 予測(max_min)
            predicted = _predict(x)

            # 実値()
            actual = _actual(x)

    def _predict(self, x):
        Z = x.reshape(1, network.self_in, network.unit_amount)

        z_ = Z[-1:]
        y_ = self.network.y.eval(session=self.network.sess, feed_dict={
            network.x: Z[-1:],
            network.n_batch: 1
        })

        return y_.reshape(-1)

    def _actual(self, x, tag):
