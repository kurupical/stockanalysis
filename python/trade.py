
class TradeController
    # 株の保有状況や利益などを管理する。
    def __init__(self):

    def _load_stock(self):

class Trade:
    # チャート予測から株の売買までを行う
    def __init__(self, stock_code):

    def predict(self):

    def _buy():

    def _sell():


class Decider:
    # 株の予測、意思決定を行う
    def __init(self):


class Chart:
    # 株の時系列データを持つ
    def __init__(self, code, stock_con):
        self.code = code
        self.chart = stock_con.get_data(code).data



if __name__ = "__main__":
    print("making now!")
    stock_con = StockController()
    stock_con.load()

    for stock_obj in stock_con.stockdata:
        chart = Chart(stock_obj.data)
