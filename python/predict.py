
from sklearn.cross_validation import train_test_split

import os
import sys
import pandas as pd
import glob
import numpy as np
import tensorflow as tf
import common
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.utils import shuffle

# TEST_MODE = False
TEST_MODE = True

if TEST_MODE:
    CSV_PATH = "../dataset/debug/stock_analysis/" #テスト
else:
    CSV_PATH = "../dataset/stock_analysis/" #本番
MODEL_PATH = "../model/GNUexport/"
# OUTPUT_ITEMで出力する項目を先頭に記述
INPUT_ITEMS = ["終値"]
OUTPUT_ITEMS = ["終値"]


class Network:
    def __init__(self, clf):
        '''
            clf : ネットワークの種類(RNN/LSTM/GRU)
        '''
        self.clf = clf

    def inference(self, x, n_batch=None, maxlen=None, n_hidden=None, n_out=None):
        # Network全体の設定を行い、モデルの出力・予想結果をかえす
        def _weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(initial)

        def _bias_variable(shape):
            initial = tf.zeros(shape, dtype=tf.float32)
            return tf.Variable(initial)

        def setcell(clf):
            if clf == 'RNN':
                cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
            elif clf == 'LSTM':
                cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
            elif clf == 'GRU':
                cell = tf.contrib.rnn.GRUCell(n_hidden)
            else:
                cell = None
            return cell

        # モデルの設定
        cell = setcell(self.clf)
        initial_state = cell.zero_state(n_batch, tf.float32)

        state = initial_state
        outputs = []
        with tf.variable_scope(self.clf):
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

    def training(self, loss, learning_rate=0.01, beta1=0.9, beta2=0.999):
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
        ary = self.data_std.values
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
        pbar = tqdm(total=len(files))
        data = np.array([[]])
        for file in files:
            read_data = pd.read_csv(file)
            if (len(read_data.index) != 0):
                stock = Stock(read_data)
                self.stockdata.append(stock)
        pbar.update(1)
        pbar.close()

    def unit_data(self, unit):
        x = np.array([[[]]])
        y = np.array([[]])
        for stock_obj in self.stockdata:
            data, target = stock_obj.unit(unit)
            if len(x) == 1:
                x = np.array(data)
                y = np.array(target)
            else:
                x = np.concatenate((x, data), axis=0)
                y = np.concatenate((y, target), axis=0)

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

def run():
    # 将来的には関数化できるよう、引数っぽいものはここに全部定義しておく
    unit = 100
    if TEST_MODE:
        epochs = 5000 # test
        n_hidden = 50 # test
        batch_size = 100
    else:
        epochs = 5000 # 本番
        n_hidden = 300 # 本番
        batch_size = 300
    test_ratio = 0.9

    history = {
        'val_loss': []
    }

    stock_con = StockController()

    stock_con.load()
    X, Y = stock_con.unit_data(unit)

    GRUNetwork = Network("GRU")

    n_in = len(X[0,0])
    n_out = len(Y[0])
    N_train = int(len(X) * test_ratio)
    N_validation = len(X) - N_train

    X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X, Y, test_size=N_validation)

    x = tf.placeholder(tf.float32, shape=[None, unit, n_in])
    t = tf.placeholder(tf.float32, shape=[None, n_out])
    n_batch = tf.placeholder(tf.int32, [])

    y = GRUNetwork.inference(x, n_batch=n_batch, maxlen=unit, n_hidden=n_hidden, n_out=n_out)
    ls = GRUNetwork.loss(y, t)
    train_step = GRUNetwork.training(ls)

    init = tf.global_variables_initializer()
    saver1 = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    n_batches = N_train // batch_size


    timetotal = common.TimeMeasure()
    timelap = common.TimeMeasure()
    for epoch in range(epochs):
        X_, Y_ = shuffle(X_train, Y_train)
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
        print("epoch:", epoch," validation loss:", val_loss, "lap:", \
              int(timelap.get()), "total:", int(timetotal.get()))
        #print("W:", sess.run(V), "b:", sess.run(c))
        timelap.reset()

        save_path = MODEL_PATH + "gnu" + str(epoch) + ".ckpt"
        if epoch % 100 == 0:
            saver1.save(sess, save_path)
    save_path = MODEL_PATH + "gnu" + str(epochs) + ".ckpt"

    truncate = unit

    # test
    stock_obj = stock_con.get_data(code=1301)
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
    plt.show()

if __name__ == '__main__':
    run()
