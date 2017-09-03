import tensorflow as tf
import random
from configparser import *

class Network:
    @staticmethod
    def read_network(path_ary):
        '''
        ckptファイルからネットワークを読み込む。
        '''
        network_ary = []
        for path in path_ary:
            config = ConfigParser()
            config.read(path + "model.ckpt.ini")
            if config['param']['network_model'] == "Network_BasicRNN":
                network_ary.append(Network_BasicRNN.read_network(config=config, path=path))

        return network_ary

class Network_BasicRNN(Network):
    @staticmethod
    def read_network(config, path):
        '''
        ckptファイルからネットワークを読み込む。
        '''
        unit_amount = int(config['param']['unit_amount'])
        n_in = int(config['param']['n_in'])
        n_out = int(config['param']['n_out'])
        n_hidden = int(config['param']['n_hidden'])
        clf = config['param']['clf']
        layer = int(config['param']['layer'])
        learning_rate = float(config['param']['learning_rate'])
        key = config['param']['key']
        codes = config['param']['codes'].split(",")
        config_path = config['param']['config_path']

        network = Network_BasicRNN(unit_amount=unit_amount,
                                   n_in=n_in,
                                   n_out=n_out,
                                   n_hidden=n_hidden,
                                   clf=clf,
                                   layer=layer,
                                   learning_rate=learning_rate,
                                   key=key,
                                   config_path=config_path)
        network.load(path + "model.ckpt")
        return network

    def __init__(self, unit_amount, n_hidden, x, y, clf, layer, learning_rate, key=None, config_path=None):
        '''
            clf : ネットワークの種類(RNN/LSTM/GRU)
        '''
        # save()で使うため、引数は保存しておく
        self.unit_amount = unit_amount
        self.n_hidden = n_hidden
        self.n_in = len(x[0,0])
        self.n_out = len(y[0])
        self.clf = clf
        self.layer = layer
        self.learning_rate = learning_rate
        if key is None:
            self.key = random.random()
        else:
            self.key = key
        self.config_path = config_path

        # placeholderの宣言
        # self.x = tf.placeholder(tf.float32, shape=[None, n_in, unit_amount])
        self.x = tf.placeholder(tf.float32, shape=[None, unit_amount, self.n_in])
        self.t = tf.placeholder(tf.float32, shape=[None, self.n_out])
        self.isTraining = tf.placeholder(tf.bool)
        self.n_batch = tf.placeholder(tf.int32, [])

        # モデル、誤差関数、学習アルゴリズムの定義
        self.y = self._inference(x=self.x, clf=clf, n_batch=self.n_batch, maxlen=n_in, n_hidden=n_hidden, n_out=self.n_out, n_in=n_in, layer=layer)
        self.ls = self._loss(self.y, self.t)
        self.train_step = self._training(self.ls, learning_rate=learning_rate)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        tf.summary.FileWriter(config_path, self.sess.graph)


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

        #if isTraining:
        #    y = self.batch_normalization([n_hidden], y)

        return y

    def _loss(self, y, t):
        mse = tf.reduce_mean(tf.square(y - t))
        return mse

    def _training(self, loss, learning_rate, beta1=0.9, beta2=0.999):
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

    def save(self, path, code_ary):
        #save_sessionと置き換える
        self.saver.save(self.sess, path)
        # configファイルを作成 https://docs.python.jp/3/library/configparser.html
        config = ConfigParser()
        codes = ""
        for code in code_ary:
            codes = codes + str(code) + ","
        config['param'] = { 'unit_amount':      self.unit_amount,
                            'n_hidden':         self.n_hidden,
                            'n_in':             self.n_in,
                            'n_out':            self.n_out,
                            'clf':              self.clf,
                            'layer':            self.layer,
                            'learning_rate':    self.learning_rate,
                            'key':              self.key,
                            'codes':            codes,
                            'config_path':      self.config_path,
                            'network_model':    self.__class__.__name__  }
        with open(path + '.ini', 'w') as configfile:
            config.write(configfile)

    def load(self, path):
        self.saver.restore(self.sess, path)

class Network_BasicRNN_SoftMax(Network):
    @staticmethod
    def read_network(config, path):
        '''
        ckptファイルからネットワークを読み込む。
        '''
        unit_amount = int(config['param']['unit_amount'])
        n_in = int(config['param']['n_in'])
        n_out = int(config['param']['n_out'])
        n_hidden = int(config['param']['n_hidden'])
        clf = config['param']['clf']
        layer = int(config['param']['layer'])
        learning_rate = float(config['param']['learning_rate'])
        key = config['param']['key']
        codes = config['param']['codes'].split(",")
        config_path = config['param']['config_path']

        network = Network_BasicRNN(unit_amount=unit_amount,
                                   n_in=n_in,
                                   n_out=n_out,
                                   n_hidden=n_hidden,
                                   clf=clf,
                                   layer=layer,
                                   learning_rate=learning_rate,
                                   key=key,
                                   config_path=config_path)
        network.load(path + "model.ckpt")
        return network

    def __init__(self, unit_amount, n_hidden, x, y, clf, layer, learning_rate, key=None, config_path=None):
        '''
            clf : ネットワークの種類(RNN/LSTM/GRU)
        '''
        # save()で使うため、引数は保存しておく
        self.unit_amount = unit_amount
        self.n_hidden = n_hidden
        self.n_in = len(x[0,0])
        self.n_out = len(y[0])
        self.clf = clf
        self.layer = layer
        self.learning_rate = learning_rate
        if key is None:
            self.key = random.random()
        else:
            self.key = key
        self.config_path = config_path

        # placeholderの宣言
        # self.x = tf.placeholder(tf.float32, shape=[None, n_in, unit_amount])
        self.x = tf.placeholder(tf.float32, shape=[None, unit_amount, self.n_in])
        self.t = tf.placeholder(tf.float32, shape=[None, self.n_out])
        self.isTraining = tf.placeholder(tf.bool)
        self.n_batch = tf.placeholder(tf.int32, [])

        # モデル、誤差関数、学習アルゴリズムの定義
        self.y = self._inference(x=self.x, clf=clf, n_batch=self.n_batch, maxlen=self.n_in, n_hidden=n_hidden, n_out=self.n_out, n_in=self.n_in, layer=layer)
        self.ls = self._loss(self.y, self.t)
        self.train_step = self._training(self.ls, learning_rate=learning_rate)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        tf.summary.FileWriter(config_path, self.sess.graph)


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
        y = tf.nn.softmax(tf.matmul(output, V) + c)

        #if isTraining:
        #    y = self.batch_normalization([n_hidden], y)

        return y

    def _loss(self, y, t):
        point = tf.constant([float(-1), float(0), float(1)], name="point")
        mse = tf.reduce_mean(tf.square(y * point - t))
        return mse

    def _training(self, loss, learning_rate, beta1=0.9, beta2=0.999):
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

    def save(self, path, code_ary):
        #save_sessionと置き換える
        self.saver.save(self.sess, path)
        # configファイルを作成 https://docs.python.jp/3/library/configparser.html
        config = ConfigParser()
        codes = ""
        for code in code_ary:
            codes = codes + str(code) + ","
        config['param'] = { 'unit_amount':      self.unit_amount,
                            'n_hidden':         self.n_hidden,
                            'n_in':             self.n_in,
                            'n_out':            self.n_out,
                            'clf':              self.clf,
                            'layer':            self.layer,
                            'learning_rate':    self.learning_rate,
                            'key':              self.key,
                            'codes':            codes,
                            'config_path':      self.config_path,
                            'network_model':    self.__class__.__name__  }
        with open(path + '.ini', 'w') as configfile:
            config.write(configfile)

    def load(self, path):
        self.saver.restore(self.sess, path)
