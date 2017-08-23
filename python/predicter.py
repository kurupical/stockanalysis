from configparser import *
from network import *
from learn import *


class Predicter:
    # 予想機
    def __init__(self, path_ary):
        self.network_ary = []
        for path in path_ary:
            # パラメータファイルを読む
            config = ConfigParser()
            config.read(path + "model.ckpt.ini")
            self.unit_amount = int(config['param']['unit_amount'])
            self.n_in = int(config['param']['n_in'])
            self.n_out = int(config['param']['n_out'])
            self.n_hidden = int(config['param']['n_hidden'])
            self.clf = config['param']['clf']
            self.layer = int(config['param']['layer'])
            self.learning_rate = float(config['param']['learning_rate'])
            self.key = config['param']['key']
            self.codes = config['param']['codes'].split(",")
            self.config_path = config['param']['config_path']

            network = Network_BasicRNN(unit_amount=self.unit_amount,
                                       n_in=self.n_in,
                                       n_out=self.n_out,
                                       n_hidden=self.n_hidden,
                                       clf=self.clf,
                                       layer=self.layer,
                                       learning_rate=self.learning_rate,
                                       key=self.key,
                                       config_path=self.config_path)
            network.load(path + "model.ckpt")
            self.network_ary.append(network)

class Predicter_Normal(Predicter):
    def __init__(self, path_ary):
        super().__init__(path_ary)

    def predict(self, charts, code, predict_term=30):
        self.original = charts.df_data['終値'].values
        self.predicted = []
        self.chart = charts
        Z = self.original[:self.unit_amount].reshape(1, self.unit_amount, self.n_in)
        for network in self.network_ary:
            predicted = []
            for i in range(predict_term):
                z_ = Z[-1:]
                y_ = network.y.eval(session=network.sess, feed_dict={
                    network.x: Z[-1:],
                    network.n_batch: 1
                    })

                seq = np.concatenate(
                    (z_.reshape(self.unit_amount, self.n_in)[1:], y_.reshape(1, self.n_in)), axis=0)

                predicted.append(y_.reshape(-1))
                seq = seq.reshape(-1, self.unit_amount, self.n_in)

                Z = np.copy(seq)
            self.predicted.append(predicted)

class Predicter_Nto1Predict_MaxMin(Predicter):
    '''
    複数銘柄の値動きから1銘柄の未来N日間の終値の最大値、最小値を予想する
    '''
    def __init__(self, path_ary):
        super().__init__(path_ary)

    def predict(self, charts, code, predict_term=None):
        input_data = []
        predicted = []
        for chart in charts:
            data = chart.df_data['終値'].values
            input_data.append(data[:self.unit_amount])
            # 予想するチャート
            if chart.code == code:
                self.chart = chart

        # 予想する銘柄の値動き
        self.original = self.chart.df_data['終値'].values
        self.original = self.original[:self.unit_amount].reshape(self.unit_amount)
        input_data = np.array(input_data)
        Z = input_data.reshape(1, self.n_in, self.unit_amount)

        for network in self.network_ary:
            z_ = Z[-1:]
            y_ = network.y.eval(session=network.sess, feed_dict={
                network.x: Z[-1:],
                network.n_batch: 1
            })
            predicted = y_.reshape(-1)
        self.predicted = predicted
