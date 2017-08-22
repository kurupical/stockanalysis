
from sklearn.cross_validation import train_test_split

import os
import sys
import pandas as pd
import glob
import numpy as np
import tensorflow as tf
import common
import random
import matplotlib.pyplot as plt
import time
import copy
import configparser

from tensorflow.contrib import rnn
from tqdm import tqdm
from sklearn.utils import shuffle

# TEST_MODE = False
TEST_MODE = True

if TEST_MODE:
    CSV_PATH = "../dataset/debug/stock_analysis/" #テスト
else:
    CSV_PATH = "../dataset/stock_analysis/" #本番
MODEL_PATH = "../model/GNUexport/"
GRAPH_PATH = "../graph/" + time.ctime().replace(" ", "_") + "/"
LOG_PATH = "../log/"
# OUTPUT_ITEMで出力する項目を先頭に記述
INPUT_ITEMS = ["終値"]
OUTPUT_ITEMS = ["終値"]
ANALYSIS_CODE = 1301
LEARNING_RATE = 0.001
UNIT = 100
TEST_EPOCHS = 500
TEST_N_HIDDEN = 300
TEST_BATCH_SIZE = 40
EPOCHS = 5000
N_HIDDEN = 300
BATCH_SIZE = 100
IS_STD_MODE = True
IS_UPDOWNRATIO_MODE = False


class Network:
    def __init__(self, unit, n_hidden, n_in, n_out, clf, layer, learning_rate, key=None):
        '''
            clf : ネットワークの種類(RNN/LSTM/GRU)
        '''
        # save()で使うため、引数は保存しておく
        self.unit = unit
        self.n_hidden = n_hidden
        self.n_in = n_in
        self.n_out = n_out
        self.clf = clf
        self.layer = layer
        self.learning_rate = learning_rate
        if key is None:
            self.key = random.random()
        else:
            self.key = key

        # placeholderの宣言
        self.x = tf.placeholder(tf.float32, shape=[None, unit, n_in])
        self.t = tf.placeholder(tf.float32, shape=[None, n_out])
        self.isTraining = tf.placeholder(tf.bool)
        self.n_batch = tf.placeholder(tf.int32, [])

        # モデル、誤差関数、学習アルゴリズムの定義
        self.y = self._inference(x=self.x, clf=clf, n_batch=self.n_batch, maxlen=unit, n_hidden=n_hidden, n_out=n_out, n_in=n_in, layer=layer)
        self.ls = self._loss(self.y, self.t)
        self.train_step = self._training(self.ls, learning_rate=learning_rate)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        tf.summary.FileWriter(LOG_PATH, self.sess.graph)


    def _inference(self, x, clf, n_batch=None, maxlen=None, n_hidden=None, n_out=None, n_in=None, layer=None, isTraining=False):
        # Network全体の設定を行い、モデルの出力・予想結果をかえす
        def _weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(initial)

        def _bias_variable(shape):
            initial = tf.zeros(shape, dtype=tf.float32)
            return tf.Variable(initial)

        def setcell(clf):
            stacked_rnn = []
            for i in range(layer):
                # multi_cell = tf.contrib.rnn.MultiRNNCell([cell] * 2)
                if clf == 'RNN':
                    cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
                elif clf == 'LSTM':
                    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
                elif clf == 'GRU':
                    cell = tf.contrib.rnn.GRUCell(n_hidden)
                else:
                    cell = None
                stacked_rnn.append(cell)
            multi_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)
            initial_state = cell.zero_state(n_batch, tf.float32)
            return multi_cell

        # モデルの設定
        cell = setcell(clf)
        initial_state = cell.zero_state(n_batch, tf.float32)

        state = initial_state
        outputs = []
        #with tf.variable_scope(clf + str(random.random())):
        with tf.variable_scope(clf, self.key):
            for t in range(maxlen):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(x[:, t, :], state)
                outputs.append(cell_output)
        output = outputs[-1]
        V = _weight_variable([n_hidden, n_out])
        c = _bias_variable([n_out])
        y = tf.matmul(output, V) + c

        '''
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, n_in])
        x = tf.split(x, maxlen, 0)

        cell_forward = setcell(clf)
        initial_state = cell_forward.zero_state(n_batch, tf.float32)
        cell_backward = setcell(clf)
        initial_state = cell_backward.zero_state(n_batch, tf.float32)

        outputs, _, _ = \
            rnn.static_bidirectional_rnn(cell_forward, cell_backward, x,
                                         dtype=tf.float32)
        V = _weight_variable([n_hidden*2, n_out])
        '''

        #if isTraining:
        #    y = self.batch_normalization([n_hidden], y)

        return y

    def _loss(self, y, t):
        mse = tf.reduce_mean(tf.square(y - t))
        return mse

    def _training(self, loss, learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999):
        optimizer = \
            tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)

        train_step = optimizer.minimize(loss)
        return train_step

    def batch_normalization(self, shape, x):
        eps = 1e-8
        beta = tf.Variable(tf.zeros(shape))
        gamma = tf.Variable(tf.ones(shape))
        mean, var = tf.nn.moments(x, [0])
        return gamma * (x - mean) / tf.sqrt(var + eps) + beta

    def save(self, path):
        #save_sessionと置き換える
        self.saver.save(self.sess, path)
        # configファイルを作成 https://docs.python.jp/3/library/configparser.html
        config = configparser.ConfigParser()
        config['param'] = { 'unit':             self.unit,
                            'n_hidden':         self.n_hidden,
                            'n_in':             self.n_in,
                            'n_out':            self.n_out,
                            'clf':              self.clf,
                            'layer':            self.layer,
                            'learning_rate':    self.learning_rate,
                            'key':              self.key }
        with open('test.ini', 'w') as configfile:
            config.write(configfile)

    def load(self, path):
        self.saver.restore(self.sess, path)



class Stock:
    def __init__(self, read_data, isStdmode=IS_STD_MODE, isUpdownratiomode=IS_UPDOWNRATIO_MODE):
        self.isStdmode = isStdmode
        try:
            codes = read_data["証券コード"].values
            self.code = codes[0]
        except KeyError: # 株以外のデータをテストデータとして使用するとき
            print("KeyError: DEBUG_MODE")
            global INPUT_ITEMS
            global OUTPUT_ITEMS
            INPUT_ITEMS=['0']
            OUTPUT_ITEMS=['0']
            self.code = ANALYSIS_CODE
        self.all_data = read_data
        data = read_data[INPUT_ITEMS]

        # 標準化データ(平均=0,標準偏差=1)
        self.data = data

        if isStdmode:
            self.stdconv = StdConverter(self.data)
            self.data = self.stdconv.data_std

        if isUpdownratiomode:
            for str in INPUT_ITEMS:
                w_ary = np.copy(data[str].values)
                ary = []
                for i in range(len(w_ary)):
                    if i == 0:
                        continue
                    if i == 1:
                        ary.append(1)
                        ary.append(w_ary[i-1] / w_ary[i])
                    if i > 1:
                        ary.append(w_ary[i-1] / w_ary[i])
                self.data[str] = ary

        for str in INPUT_ITEMS:
            self.all_data[str] = self.data[str]

    def unit(self, unit):
        x = np.array([[[]]])
        y = np.array([[]])

        data = []
        target = []
        # ary = self.data[-unit*2:].values
        ary = self.data.values
        if len(self.data) > unit:
            for i in range(0, len(ary) - unit):
                data.append(ary[i:i + unit, :])
                target.append(ary[i + unit, :len(OUTPUT_ITEMS)])
            if len(x) == 1:
                x = np.array(data).reshape(len(data), unit, len(data[0][0]))
                y = np.array(target).reshape(len(target),len(target[0]))
            else:
                x = np.concatenate((x, np.array(data).reshape(len(data), unit, len(data[0][0]))), axis=0)
                y = np.concatenate((y, np.array(target).reshape(len(data), len(target[0]))), axis=0)
        return x, y

    # def convertupdownratio(self, ary):

    def get_index(self, item_name):
        if len(item_name) == 1:
            return INPUT_ITEMS.index(item_name[0])
        else:
            # 未実装
            return None


class StockController:
    def __init__(self):
        self.stockdata = [] # Stockオブジェクトを格納するlist

    def load(self):
        input_path = CSV_PATH + "*.csv"
        files = glob.glob(input_path)
        print ("load")
        pbar = tqdm(total=len(files))
        data = np.array([[]])
        for file in files:
            read_data = pd.read_csv(file)
            if (len(read_data.index) != 0):
                stock = Stock(read_data, isUpdownratiomode=False)
                self.stockdata.append(stock)
            pbar.update(1)
        pbar.close()

    def search_high_cor(self, cor, code, unit):
        '''
         指定した銘柄の最新からunit日前〜0日前のデータと相関の高い
         「最新からunit*2日前〜unit+1日前の銘柄」をstockdataにセットする
         (指定されたコードもセットする)
         param
            cor : 基準となる相関係数
            code : 指定する銘柄
            unit : １銘柄あたりのデータの単位(unit日分)
        '''
        ary = []
        x = self.get_data(code)
        x = x.data[-unit:]
        print ("search_high_cor")
        amount_of_search = len(self.stockdata)
        pbar = tqdm(total=len(self.stockdata))
        for stock_obj in self.stockdata:
            if stock_obj.code == code:
                stock_obj.data = stock_obj.data[-unit*2:]
                ary.append(stock_obj)
            if len(stock_obj.data) > unit*2:
                y = stock_obj.data[-unit*2:-unit]
                xy_cor = np.corrcoef(x.values.reshape(-1), y.values.reshape(-1))[0][1]
                if abs(xy_cor) > abs(cor):
                    stock_obj.data = stock_obj.data[-unit*2:]
                    ary.append(stock_obj)
                    print("証券コード:", stock_obj.code, " 相関係数:", xy_cor)
            pbar.update(1)
        pbar.close()
        self.stockdata = ary
        print("search_high_cor 結果:\n")
        print("*******************************************")
        print("分析銘柄数:", amount_of_search)
        print("高相関銘柄:", len(self.stockdata))
        print("*******************************************")


    def unit_data(self, unit):
        x = np.array([[[]]])
        y = np.array([[]])
        print ("unit_data")
        pbar = tqdm(total=len(self.stockdata))
        for stock_obj in self.stockdata:
            data, target = stock_obj.unit(unit)
            if len(x) == 1:
                x = np.array(data)
                y = np.array(target)
            else:
                x = np.concatenate((x, data), axis=0)
                y = np.concatenate((y, target), axis=0)
            pbar.update(1)
        pbar.close()

        print("unit_data 結果:\n")
        print("*******************************************")
        print("分析銘柄数:", len(self.stockdata))
        print("入力データ:", len(x))
        print("*******************************************")

        return x, y

    def get_data(self, code):
        for i in range(len(self.stockdata)):
            stock_obj = self.stockdata[i]
            if stock_obj.code == code:
                return stock_obj

class StdConverter:
    '''
        データ群の標準化、標準化戻しを行う
        attribute
            numpy/pandas data : 標準化対象のデータ
            pandas _std_info : dataの配列ごとの平均、標準偏差を格納
    '''
    def __init__(self, data):
        self.data = data
        self.data_type = type(data)

        self.std(self.data)

    def std(self, data=None):
        self.data_std = data
        self._std_info = pd.DataFrame()

        # numpy/pandas
        if type(data) == type(pd.DataFrame()):
            obj_type = "pd"
            self.length = len(data.columns)
            data = data.values
        elif type(data) == type(np.array([[]])):
            obj_type = "np"
            self.length = len(data[0])
            data = data

        for i in range(self.length):
            ary = np.copy(data[:,i])
            ary_std = (ary - ary.mean()) / ary.std()
            if obj_type == "pd":
                self.data_std.iloc[:, i] = ary_std
            elif obj_type == "np":
                self.data_std[:, i] = ary_std
            std_info = pd.Series([i, ary.mean(), ary.std()], index=['item' ,'mean' ,'std'])
            self._std_info = self._std_info.append(std_info, ignore_index=True)

    def unstd(self, data=None):
        # numpy/pandas
        if type(data) == type(pd.DataFrame()):
            data = data.values
        elif type(data) == type(np.array([[]])):
            data = data

        data_unstd = np.copy(data)
        for i in range(self.length):
            df = self._std_info[self._std_info["item"] == i]
            data_unstd[:] = data_unstd[:] * df['std'].values + df['mean'].values
        return data_unstd


def run(unit, epochs, n_hidden, learning_rate, batch_size, clf, layer, stock_con, name=""):
    # 将来的には関数化できるよう、引数っぽいものはここに全部定義しておく
    def save_pred_graph():
        # test
        stock_obj = stock_con.get_data(code=ANALYSIS_CODE)
        original = stock_obj.data.values
        Z = original[:unit].reshape(1, unit, n_in)
        predicted = []

        stdconv = copy.deepcopy(stock_obj.stdconv)
        for i in range(len(original) - unit):
            z_ = Z[-1:]
            y_ = network.y.eval(session=network.sess, feed_dict={
                network.x: Z[-1:],
                network.n_batch: 1
            })

            seq = np.concatenate(
                (z_.reshape(unit, n_in)[1:], y_.reshape(1, n_in)), axis=0)
            '''
            # 標準化を元に戻して、再度標準化する
            seq = stdconv.unstd(seq)
            stdconv = StdConverter(seq)
            seq = stdconv.data_std
            '''
            predicted.append(y_.reshape(-1))
            seq = seq.reshape(-1, unit, n_in)

            Z = np.copy(seq)

        if stock_obj.isStdmode:
            original_endval = stock_obj.stdconv.unstd(original)
        else:
            original_endval = stock_obj.data.values

        predicted = np.array(predicted)
        predicted = stock_obj.stdconv.unstd(predicted)
        teachdata_endval = original_endval[:unit, stock_obj.get_index(INPUT_ITEMS)]
        predict_endval = np.append(teachdata_endval, predicted[:, stock_obj.get_index(INPUT_ITEMS)], axis=0)

        plt.rc('font', family='serif')
        plt.figure()
        plt.plot(original_endval, linestyle='dotted', color='#aaaaaa')
        plt.plot(teachdata_endval, linestyle='dashed', color='black')
        plt.plot(predict_endval, color='black')
        filename = GRAPH_PATH + "pred/" + get_filename() + ".png"
        plt.savefig(filename)

    def save_loss_graph():
        plt.rc('font', family="serif")
        plt.figure()
        plt.plot(history['val_loss'], color='black')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        filename = GRAPH_PATH + "loss/" + get_filename() + ".png"
        plt.savefig(filename)

    def save_session():
        save_path = MODEL_PATH + get_filename() + ".ckpt"
        network.save(save_path)

    def get_filename():
        return name + "loss" + str(round(val_loss,3)) + "epoch" + str(epoch) + "★UNIT:" + str(unit) + "-HID:" + str(n_hidden) + "-lr:" + str(learning_rate) + "-clf:" + clf + "-layer:" + str(layer)

    if not os.path.isdir(GRAPH_PATH):
        os.mkdir(GRAPH_PATH)
        os.mkdir(GRAPH_PATH + "loss/")
        os.mkdir(GRAPH_PATH + "pred/")
    test_ratio = 0.7
    history = {
        'val_loss': []
    }

    if len(stock_con.stockdata) == 0:
        #初回だけデータロード
        stock_con.load()
        stock_con.search_high_cor(cor=0.6, code=ANALYSIS_CODE, unit=unit)

    X, Y = stock_con.unit_data(unit)

    n_in = len(X[0,0])
    n_out = len(Y[0])
    N_train = int(len(X) * test_ratio)
    N_validation = len(X) - N_train

    network = Network(unit=unit, n_hidden=n_hidden, n_in=n_in, n_out=n_out, clf=clf, layer=layer, learning_rate=learning_rate)

    X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X, Y, test_size=N_validation)

    n_batches = N_train // batch_size

    timetotal = common.TimeMeasure()
    timelap = common.TimeMeasure()
    loss_ary = []
    for epoch in range(epochs):
        X_, Y_ = shuffle(X_train, Y_train, random_state=0)
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            network.sess.run(network.train_step, feed_dict={
                network.x : X_[start:end],
                network.t : Y_[start:end],
                network.n_batch: batch_size,
                network.isTraining : True
            })

        val_loss = network.ls.eval(session=network.sess, feed_dict={
            network.x: X_validation,
            network.t: Y_validation,
            network.n_batch: N_validation,
            network.isTraining : True
        })

        history['val_loss'].append(val_loss)
        run_log = "epoch:" + str(epoch) + " validation loss:" + str(val_loss) + " lap:" + \
                  str(round(timelap.get(), 2)) + " total:" + str(round(timetotal.get(), 2))
        print(run_log)
        #print("W:", sess.run(V), "b:", sess.run(c))
        timelap.reset()

        if (epoch+1) % 1000 == 0:
            save_pred_graph()

        if (epoch+1) % 1000 == 0:
            save_session()

        if (epoch+1) % 10000 == 0:
            save_loss_graph()
    save_pred_graph()
    save_session()



if __name__ == '__main__':

    unit = [100]
    learning_rate = [0.001]
    n_hidden = [30]
    classifier = ["GRU"]
    layer = [1]
    epochs = 100000
    stock_con = StockController()
    for un in unit:
        for lr in learning_rate:
            for hid in n_hidden:
                for clf in classifier:
                    for ly in layer:
                        run(unit=un, \
                            learning_rate=lr, \
                            n_hidden=hid, \
                            epochs=epochs, \
                            clf=clf, \
                            batch_size=50, \
                            layer=ly, \
                            stock_con=stock_con)
