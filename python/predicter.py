from configparser import *
from learn import *

class Predicter:
    # 予想機
    def __init__(self, path_ary):
        self.network_ary = []
        for path in path_ary:
            # パラメータファイルを読む
            config = ConfigParser()
            config.read(path + "model.ckpt.ini")
            self.unit = int(config['param']['unit_amount'])
            self.n_in = int(config['param']['n_in'])
            self.n_out = int(config['param']['n_out'])
            self.n_hidden = int(config['param']['n_hidden'])
            self.clf = config['param']['clf']
            self.layer = int(config['param']['layer'])
            self.learning_rate = float(config['param']['learning_rate'])
            self.key = config['param']['key']

            network = Network(unit=self.unit,
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
