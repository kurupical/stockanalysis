
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

from tqdm import tqdm
from sklearn.utils import shuffle

TEST_MODE = False
# TEST_MODE = True

if TEST_MODE:
    CSV_PATH = "../dataset/debug/stock_analysis/" #テスト
else:
    CSV_PATH = "../dataset/stock_analysis/" #本番
MODEL_PATH = "../model/GNUexport/"
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


class Network:
    def __init__(self, clf):
        '''
            clf : ネットワークの種類(RNN/LSTM/GRU)
        '''
        self.clf = clf

    def inference(self, x, n_batch=None, maxlen=None, n_hidden=None, n_out=None, layer=None):
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
            return multi_cell

        # モデルの設定
        cell = setcell(self.clf)
        initial_state = cell.zero_state(n_batch, tf.float32)

        state = initial_state
        outputs = []
        with tf.variable_scope(self.clf + str(random.random())):
            for t in range(maxlen):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(x[:, t, :], state)
                outputs.append(cell_output)

        output = outputs[-1]

        V = _weight_variable([n_hidden, n_out])
        c = _bias_variable([n_out])
        y = tf.matmul(output, V) + c

        return y

    def loss(self, y, t):
        mse = tf.reduce_mean(tf.square(y - t))
        return mse

    def training(self, loss, learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999):
        optimizer = \
            tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)

        train_step = optimizer.minimize(loss)
        return train_step


class Stock:
    def __init__(self, read_data):
        codes = read_data["証券コード"].values
        self.code = codes[0]
        self.data_std_info = pd.DataFrame(columns=["item", "mean", "std"])
        data = read_data[INPUT_ITEMS]


        # 標準化データ(平均=0,標準偏差=1)
        self.data_std = data
        for str in INPUT_ITEMS:
            ary = np.copy(data[str].values)
            self.data_std[str] = (ary - ary.mean()) / ary.std()
            std_info = pd.Series([str, ary.mean(), ary.std()], index=self.data_std_info.columns)
            self.data_std_info = self.data_std_info.append(std_info, ignore_index=True)

    def unit(self, unit):
        x = np.array([[[]]])
        y = np.array([[]])

        data = []
        target = []
        ary = self.data_std[-unit*2:].values
        if len(self.data_std) > unit:
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

    def unstd(self, data=None):
        if data is None:
            data_unstd = np.copy(self.data_std)
        else:
            data_unstd = np.copy(data)

        for i, str in zip(range(len(INPUT_ITEMS)), INPUT_ITEMS):
            df = self.data_std_info[self.data_std_info["item"] == str]
            data_unstd[:,i] = data_unstd[:,i] * df['std'].values + df['mean'].values

        return data_unstd

    def get_index(self, item_name):
        return INPUT_ITEMS.index(item_name)


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
                stock = Stock(read_data)
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
        x = x.data_std[-unit:]
        print ("search_high_cor")
        amount_of_search = len(self.stockdata)
        pbar = tqdm(total=len(self.stockdata))
        for stock_obj in self.stockdata:
            if stock_obj.code == code:
                ary.append(stock_obj)
            else:
                if len(stock_obj.data_std) > unit*2:
                    y = stock_obj.data_std[-unit*2:-unit]
                    xy_cor = np.corrcoef(x.values.reshape(-1), y.values.reshape(-1))[0][1]
                    if abs(xy_cor) > cor:
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

def run(unit, epochs, n_hidden, learning_rate, batch_size, clf, layer, stock_con):
    # 将来的には関数化できるよう、引数っぽいものはここに全部定義しておく
    test_ratio = 0.9

    history = {
        'val_loss': []
    }

    if len(stock_con.stockdata) == 0:
        #初回だけデータロード
        stock_con.load()
        stock_con.search_high_cor(cor=0.5, code=ANALYSIS_CODE, unit=unit)

    X, Y = stock_con.unit_data(unit)

    network = Network(clf)

    n_in = len(X[0,0])
    n_out = len(Y[0])
    N_train = int(len(X) * test_ratio)
    N_validation = len(X) - N_train

    X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X, Y, test_size=N_validation)

    x = tf.placeholder(tf.float32, shape=[None, unit, n_in])
    t = tf.placeholder(tf.float32, shape=[None, n_out])
    n_batch = tf.placeholder(tf.int32, [])

    y = network.inference(x, n_batch=n_batch, maxlen=unit, n_hidden=n_hidden, n_out=n_out, layer=layer)
    ls = network.loss(y, t)
    train_step = network.training(ls, learning_rate=learning_rate)

    init = tf.global_variables_initializer()
    saver1 = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    n_batches = N_train // batch_size


    timetotal = common.TimeMeasure()
    timelap = common.TimeMeasure()
    for epoch in range(epochs):
        X_, Y_ = shuffle(X_train, Y_train, random_state=0)
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            sess.run(train_step, feed_dict={
                x : X_[start:end],
                t : Y_[start:end],
                n_batch: batch_size
            })

        val_loss = ls.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation,
            n_batch: N_validation
        })

        history['val_loss'].append(val_loss)
        run_log = "epoch:" + str(epoch) + " validation loss:" + str(val_loss) + " lap:" + \
                  str(round(timelap.get(), 2)) + " total:" + str(round(timetotal.get(), 2))
        print(run_log)
        #print("W:", sess.run(V), "b:", sess.run(c))
        timelap.reset()

        if epoch+1 % 1000 == 0:
            save_path = MODEL_PATH + "loss" + str(val_loss) + "epoch" + str(epoch) + "UNIT" + str(unit) + "-N_HIDDEN" + str(n_hidden) + "-learning_rate" + str(learning_rate) + "-clf" + clf + "-layer" + str(layer) + ".ckpt"
            saver1.save(sess, save_path)
    save_path = MODEL_PATH + "loss" + str(val_loss) + "epoch" + str(epoch) + "UNIT" + str(unit) + "-N_HIDDEN" + str(n_hidden) + "-learning_rate" + str(learning_rate) + "-clf" + clf + "-layer" + str(layer) + ".ckpt"
    saver1.save(sess, save_path)

    truncate = unit

    # test
    stock_obj = stock_con.get_data(code=ANALYSIS_CODE)
    original = stock_obj.unstd()
    Z = original[:unit].reshape(1, unit, n_in)
    predicted = []

    for i in range(len(original) - unit):
        z_ = Z[-1:]
        y_ = y.eval(session=sess, feed_dict={
            x: Z[-1:],
            n_batch: 1
        })

        seq = np.concatenate(
            (z_.reshape(unit, n_in)[1:], y_.reshape(1, n_in)), axis=0)\
            .reshape(1, unit, n_in)

        # Z = np.append(Z, seq, axis=0)
        Z = seq
        predicted.append(y_.reshape(-1))

    predicted = np.array(predicted)
    predicted = stock_obj.unstd(predicted)

    # よーわからんけど、originalが上書きされるので間抜けだけどいったんこれで・・・
    original_endval = original[:, stock_obj.get_index("終値")]
    teachdata_endval = original[:unit, stock_obj.get_index("終値")]
    #original_endval = 10**stock.get(code="1301")[:, 0]
    #teachdata_endval = 10**stock.get(code="1301")[:unit, 0]
    predict_endval = np.append(teachdata_endval, predicted[:, stock_obj.get_index("終値")], axis=0)

    plt.rc('font', family='serif')
    plt.figure()
    plt.plot(original_endval, linestyle='dotted', color='#aaaaaa')
    plt.plot(teachdata_endval, linestyle='dashed', color='black')
    plt.plot(predict_endval, color='black')
    filename = "loss" + str(val_loss) + "★UNIT" + str(unit) + "-N_HIDDEN" + str(n_hidden) + "-learning_rate" + str(learning_rate) + "-clf" + clf + "-layer" + str(layer) + ".png"
    plt.savefig(filename)
    # plt.show()

if __name__ == '__main__':

    unit = [50]
    learning_rate = [0.001]
    n_hidden = [50, 100]
    classifier = ["LSTM", "GRU"]
    layer = [2,3]
    stock_con = StockController()
    for un in unit:
        for lr in learning_rate:
            for hid in n_hidden:
                for clf in classifier:
                    for ly in layer:
                        run(unit=un, \
                            learning_rate=lr, \
                            n_hidden=hid, \
                            epochs=5000, \
                            clf=clf, \
                            batch_size=40, \
                            layer=ly, \
                            stock_con=stock_con)
