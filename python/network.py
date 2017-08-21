import tensorflow as tf
import random
import configparser

class Network_BasicRNN:
    def __init__(self, unit_amount, n_hidden, n_in, n_out, clf, layer, learning_rate, key=None, config_path=None):
        '''
            clf : ネットワークの種類(RNN/LSTM/GRU)
        '''
        # save()で使うため、引数は保存しておく
        self.unit_amount = unit_amount
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
        self.x = tf.placeholder(tf.float32, shape=[None, unit_amount, n_in])
        self.t = tf.placeholder(tf.float32, shape=[None, n_out])
        self.isTraining = tf.placeholder(tf.bool)
        self.n_batch = tf.placeholder(tf.int32, [])

        # モデル、誤差関数、学習アルゴリズムの定義
        self.y = self._inference(x=self.x, clf=clf, n_batch=self.n_batch, maxlen=unit_amount, n_hidden=n_hidden, n_out=n_out, n_in=n_in, layer=layer)
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

    def save(self, path):
        #save_sessionと置き換える
        self.saver.save(self.sess, path)
        # configファイルを作成 https://docs.python.jp/3/library/configparser.html
        config = configparser.ConfigParser()
        config['param'] = { 'unit_amount':      self.unit_amount,
                            'n_hidden':         self.n_hidden,
                            'n_in':             self.n_in,
                            'n_out':            self.n_out,
                            'clf':              self.clf,
                            'layer':            self.layer,
                            'learning_rate':    self.learning_rate,
                            'key':              self.key }
        with open(path + '.ini', 'w') as configfile:
            config.write(configfile)

    def load(self, path):
        self.saver.restore(self.sess, path)
