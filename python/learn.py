# stockanalysis library
from common import *
from configuration import *
from logger import *
from verify_model import *
# common library
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import os

class Learn:
    def __init__(self, stock_con, network, test_ratio, unit_amount, batch_size, result_path, code_ary, verify_model):
        self.x = stock_con.data_x
        self.y = stock_con.data_y
        self.stock_con = stock_con
        self.network = network
        self.test_ratio = test_ratio
        self.unit_amount = unit_amount
        self.batch_size = batch_size
        self.verify_model = verify_model
        self.n_in = len(self.x[0])
        self.n_out = len(self.y[0])
        self.N_train = int(len(self.x) * test_ratio)
        self.N_validation = len(self.x) - self.N_train
        self.history = {
            'val_loss': []
        }
        self.result_path = result_path
        if not os.path.isdir(self.result_path):
            os.mkdir(self.result_path)

        self.X_train, self.X_validation, self.Y_train, self.Y_validation = \
            train_test_split(self.x, self.y, test_size=self.N_validation)

        self.n_batches = self.N_train // self.batch_size
        self.code_ary = code_ary

        self.logger = Logger(Configuration.log_path)

    def learn(self, epochs):
        timetotal = TimeMeasure()
        timelap = TimeMeasure()
        loss_ary = []
        for epoch in range(epochs):
            X_, Y_ = shuffle(self.X_train, self.Y_train)
            for i in range(self.n_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                self.network.sess.run(self.network.train_step, feed_dict={
                    self.network.x : X_[start:end],
                    self.network.t : Y_[start:end],
                    self.network.n_batch: self.batch_size,
                    self.network.isTraining : True
                })

            val_loss = self.network.ls.eval(session=self.network.sess, feed_dict={
                self.network.x: self.X_validation,
                self.network.t: self.Y_validation,
                self.network.n_batch: self.N_validation,
                self.network.isTraining : True
            })

            self.history['val_loss'].append(val_loss)
            run_log = "epoch:" + str(epoch) + " validation loss:" + str(val_loss) + " lap:" + \
                  str(round(timelap.get(), 2)) + " total:" + str(round(timetotal.get(), 2))
            print(run_log)
            timelap.reset()

            if (epoch+1) % 100 == 0:
                self.logger.log(run_log)
            if (epoch+1) % 1000 == 0:
                self._output_result(epoch)

        self._output_result(epoch)
        self._output_loss_graph()

    def _output_result(self, epoch):
        path = self.result_path + "/" + str(epoch) + "/"
        if not os.path.isdir(path):
            os.mkdir(path)
        self._save(epoch, path)
        self._verify_model(epoch, path)

    def _output_loss_graph(self):
        path = self.result_path + "/loss_graph.jpeg"

        plt.figure()
        data = self.history['val_loss']
        plt.plot(data, color='black', label='loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        filename = path
        plt.savefig(filename)

    def _save(self, epoch, path):
        # ネットワーク情報
        path_model = path + "model.ckpt"
        self.network.save(path=path_model, code_ary=self.code_ary)

    def _verify_model(self, epoch, path):
        verifier = VerifyModel.generate_verify_model(network=self.network, stock_con=self.stock_con)
        verifier.verify(path=path)
