from stock import *
from unitrule_stock import *
from unitrule_stockcon import *
from network import *
from learn2 import *


def test():
    # とりあえずなんか動かしたい時用
    # @param
    unit_amount = 50
    forward_day = 30
    predict_mode = "max_min"
    # predict_mode = "normal"
    csv_path = "../dataset/debug/test/"
    test_ratio = 0.8
    batch_size = 50
    input_items = ["終値"]
    output_items = ["終値"]
    n_day = 150
    layer = 1
    n_hidden = 30
    clf = "GRU"
    learning_rate = 0.001
    key = None
    epochs = 1000
    config_path = "net_config.ini"
    result_path = "../result/" + time.ctime().replace(" ", "_") + "/"
    YMDbefore = "2016/12/31"

    unitrule_s = UnitRule_Stock_ForwardDay(unit_amount=unit_amount, forward_day=forward_day, predict_mode=predict_mode)
    unitrule_sc = UnitRule_Stockcon_Bundle()
    stock_con = StockController(csv_path=csv_path,
                                unitrule_stock=unitrule_s,
                                unitrule_stockcon=unitrule_sc,
                                input_items=input_items,
                                output_items=output_items)
    stock_con.load()
    # 学習データの絞り込み１　＠　時価総額 making now
    # stock_con.search_isinrange_marketcap(min_value=0, max_vaule=30**11)
    # 学習データの絞り込み２　学習対象の日付
    stock_con.search_is_YMDbefore(ymd=YMDbefore)
    # 学習データの絞り込み３　＠　過去N日
    stock_con.search_isexist_past_Nday(n_day=n_day)
    x, y = stock_con.unit_data()
    # networkの作成
    n_in = len(x[0][0])
    n_out = len(y[0])
    # 学習/保存
    network = Network_BasicRNN(unit_amount=unit_amount,
                               n_hidden=n_hidden,
                               n_in=n_in,
                               n_out=n_out,
                               clf=clf,
                               layer=layer,
                               learning_rate=learning_rate,
                               key=key,
                               config_path=config_path)

    learner = Learn(x=x,
                    y=y,
                    network=network,
                    test_ratio=test_ratio,
                    unit_amount=unit_amount,
                    batch_size=batch_size,
                    result_path=result_path)

    learner.learn(epochs=epochs)

def main():
    # iniファイルから読むとかいう処理はこっちで後で書く
    print("making now!")

if __name__ == "__main__":
    test()
