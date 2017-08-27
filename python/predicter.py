from configparser import *
from network import *
from learn import *


class Predicter:
    @staticmethod
    def generate_predicter(predicter_model, network_ary):
        '''
        指定されたpredicter_modelのPredicterクラスを生成する
        (あとで Predicter_Controllerを作成する(#49))
        '''
        if predicter_model == "Predicter_Normal":
            return Predicter_Normal(network_ary)
        if predicter_model == "Predicter_Nto1Predict_MaxMin":
            return Predicter_Nto1Predict_MaxMin(network_ary)

    # 予想機
    def __init__(self, network_ary):
        self.network_ary = network_ary

class Predicter_Normal(Predicter):
    def __init__(self, network_ary):
        super().__init__(network_ary)

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
    def __init__(self, network_ary):
        super().__init__(network_ary)

    def predict(self, charts, code, predict_term=None):
        input_data = []
        predicted = [[]]

        # あとで、1Predicter=1Networkにする (#48) for文書いてるけど複数networkには対応していません
        for network in self.network_ary:
            for chart in charts:
                if chart.code == code:
                    self.chart = chart
                else:
                    data = chart.df_data['終値'].values
                    input_data.append(data[:network.unit_amount])

            self.original = self.chart.df_data['終値'].values
            original = self.original[:network.unit_amount].reshape(1, -1)
            input_data = np.array(input_data)
            marge_data = np.concatenate((original, input_data), axis=0)
            # 予想対象の銘柄を先頭にする
            Z = marge_data.reshape(1, network.n_in, network.unit_amount)

            z_ = Z[-1:]
            y_ = network.y.eval(session=network.sess, feed_dict={
                network.x: Z[-1:],
                network.n_batch: 1
            })
            predicted = y_.reshape(-1)
            self.predicted = np.array(predicted).reshape(-1, network.n_out)
