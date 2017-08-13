
import learn
MODEL_PATH = "../model/GNUexport/"

class TradeController:
    # 株の保有状況や利益などを管理する。
    def __init__(self):
        print("makng now!")
    def _load_stock(self):
        print("makng now!")

class Trade:
    # チャート予測から株の売買までを行う
    def __init__(self, code, strategy, predict):
        self.code = code
        self.decider = \
            Decider(code, strategy, predict)

    def predict(self):
        print("makng now!")

    def _buy():
        print("makng now!")

    def _sell():
        print("makng now!")

class Decider:
    # 株の予測、意思決定を行う
    def __init__(self, code, strategy, predict):
        self.code = code
        self.strategy = strategy
        self.predict = predict

    def predict(self):
        print("makng now!")

class Predict:
    # 予想機
    def __init__(self, path_ary):
        self.network_ary = []
        for path in path_ary:
            # パラメータファイルを読む処理
            ~~~
            
            network = learn.Network()
            network.load(path)
            self.network_ary.append(network)


class Chart:
    # 株の時系列データを持つ
    def __init__(self, code, stock_con):
        self.code = code
        self.chart = stock_con.get_data(code).data



if __name__ == "__main__":
    print("making now!")

    path_ary = [MODEL_PATH + "loss0.086epoch99★UNIT:100-HID:30-lr:0.001-clf:GRU-layer:1.ckpt"]
    #test
    predict = Predict(path_ary=path_ary)
    #test
    stock_con = StockController()
    stock_con.load()

    for stock_obj in stock_con.stockdata:
        chart = Chart(stock_obj.data)
