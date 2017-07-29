
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
INPUT_ITEM = ["終値", "出来高", "日付", "証券コード", "売上高", "純利益", "総資産"]
INPUT_ITEM_INDEX_OF_DATE = INPUT_ITEM.index("日付")
INPUT_ITEM_INDEX_OF_CODE = INPUT_ITEM.index("証券コード")

OUTPUT_ITEM = ["終値", "出来高"]


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
    def __init__(self):
        # 株価の取得
        input_path = CSV_PATH + "*.csv"
        files = glob.glob(input_path)

        df = pd.DataFrame()
        print("Data Load:")
        pbar = tqdm(total=len(files))
        data = np.array([[]])
        for file in files:
        # pandasよりnumpyのほうが早い！
        #    read_df = pd.read_csv(file)
        #    df = pd.concat([df, read_df], axis=0)
            read_data = pd.read_csv(file)
            read_data = read_data[INPUT_ITEM].values
            if len(read_data) != 0:
                if len(data) == 1:
                    data = read_data
                else:
                    data = np.concatenate((data, read_data), axis=0)
            pbar.update(1)
        pbar.close()
        self.data = data
        self.output_index = []

    def get(self, code=None):
        # pandas -> numpyに置き換え
        # return self.df[(self.df['証券コード'] == int(code[0:4]))]
        index = np.where(self.data[:, INPUT_ITEM_INDEX_OF_CODE] == int(code[0:4]))
        data = self.data[index]
        return data[:]

    def unit_data(self, unit):
        '''
        株式データをunit単位に分割した教師データを返す。
        param:
            unit : 分割単位
                (例:unit=3)
                [1,2,3,...,99,100] ->
                    入力 : [1,2,3],[2,3,4],[3,4,5],...,[97,98,99]
                    出力 : 4,5,6,...,100
        return:
            x : 教師データの入力
            y : 教師データの出力

        '''
        code_count = 0
        data_count = 0
        x = np.array([[[]]])
        y = np.array([[]])
        codes = common.get_stockscode()
        print("Data Unit:")
        pbar2 = tqdm(total=len(codes))
        for code in codes:
        #    df = self.get(code=code)
        #    ary = df.value
            ary = self.get(code=code)
            data = []
            target = []
            if len(ary) > unit:
                code_count += 1
                data_count += len(ary)
                for i in range(0, len(ary) - unit):
                    data.append(ary[i:i + unit, :])
                    target.append(ary[i + unit, :len(OUTPUT_ITEM)])
                if len(x) == 1:
                    x = np.array(data).reshape(len(data), unit, len(data[0][0]))
                    y = np.array(target).reshape(len(target),len(target[0]))
                else:
                    x = np.concatenate((x, np.array(data).reshape(len(data), unit, len(data[0][0]))), axis=0)
                    y = np.concatenate((y, np.array(target).reshape(len(data), len(target[0]))), axis=0)
            pbar2.update(1)
        pbar2.close()
        print("getdata 結果:\n")
        print("*******************************************")
        print("分析銘柄数:", code_count)
        print("分析データ行数:", data_count)
        print("NWへの入力データ:", len(x))
        print("*******************************************")

        return x, y

def completion(data, x):
    '''
    予測データの他データを補完する。
    param:
        x : 配列[始値、高値、安値、終値、出来高]
    return:
        xに売上情報などがすべて補完されたデータ
    '''

    data_m = data[:]
    data_m[INPUT_ITEM_INDEX_OF_DATE] = np.log10(10**data[INPUT_ITEM_INDEX_OF_DATE] + 1)
    for i in range(len(x)):
        data_m[i] = x[i]
    data_m[1]

    return data_m

def run():
    # 将来的には関数化できるよう、引数っぽいものはここに全部定義しておく
    unit = 100
    if TEST_MODE:
        epochs = 100 # test
        n_hidden = 2000 # test
        batch_size = 30
    else:
        epochs = 5000 # 本番
        n_hidden = 300 # 本番
        batch_size = 300
    test_ratio = 0.9

    history = {
        'val_loss': []
    }

    stock = Stock()
    X, Y = stock.unit_data(unit)

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

    original = stock.get(code="1301")
    Z = original[:unit].reshape(1, unit, n_in)
    predicted = []

    for i in range(len(original) - unit):
        z_ = Z[-1:]
        y_ = y.eval(session=sess, feed_dict={
            x: Z[-1:],
            n_batch: 1
        })

        y_cmpl = completion(z_[0, -1], y_[0])

        seq = np.concatenate(
            (z_.reshape(unit, n_in)[1:], y_cmpl.reshape(1, n_in)), axis=0)\
            .reshape(1, unit, n_in)

        # Z = np.append(Z, seq, axis=0)
        Z = seq
        predicted.append(y_cmpl)

    predicted = np.array(predicted)

    # よーわからんけど、originalが上書きされるので間抜けだけどいったんこれで・・・
    # original_endval = 10 ** original[:, 4]
    # teachdata_endval = 10 ** original[:unit, 4]
    original_endval = stock.get(code="1301")[:, 0]
    teachdata_endval = stock.get(code="1301")[:unit, 0]
    predict_endval = np.append(teachdata_endval, predicted[:, 0], axis=0)

    plt.rc('font', family='serif')
    plt.figure()
    plt.plot(original_endval, linestyle='dotted', color='#aaaaaa')
    plt.plot(teachdata_endval, linestyle='dashed', color='black')
    plt.plot(predict_endval, color='black')
    plt.show()

if __name__ == '__main__':
    run()
