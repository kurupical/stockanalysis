# stockanalysis library
from stock import *
from unitrule_stock import *
from unitrule_stockcon import *
from network import *
from learn import *
# common library
import datetime

def test():
    # とりあえずなんか動かしたい時用
    # @param
    unit_amount = 200
    forward_day = 15
    predict_mode = "max_min_classify"
    # predict_mode = "max_min"
    # predict_mode = "normal"
    csv_path = "../dataset/debug/stock_analysis/"
    stockinfo_path = "../dataset/stock_info.csv"
    test_ratio = 0.9
    batch_size = 50
    min_value = 0
    max_value = 0.7*(10**10)
    classify_ratio = 0.1
    input_items = ["終値"]
    output_items = ["終値"]
    unitrule_stockcon = "UnitRule_Stockcon_Normal"
    n_day = 800
    layer = 4
    n_hidden = 30
    clf = "GRU"
    learning_rate = 0.001
    key = None
    epochs = 10000
    result_path = "../result/" + datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S") + "/"
    config_path = result_path + "net_config.ini"
    verify_model = "max_min_classify_verify"
    YMDbefore = "2016/12/31"

    Configuration.log_path = result_path

    unitrule_s = UnitRule_Stock_ForwardDay(unit_amount=unit_amount, forward_day=forward_day, predict_mode=predict_mode, classify_ratio=classify_ratio)
    unitrule_sc = UnitRule_Stockcon.generate_unitrule_stockcon("UnitRule_Stockcon_Normal")
    stock_info = StockInfo(path=stockinfo_path)
    stock_con = StockController(csv_path=csv_path,
                                unitrule_stock=unitrule_s,
                                unitrule_stockcon=unitrule_sc,
                                stock_info=stock_info,
                                input_items=input_items,
                                output_items=output_items)
    stock_con.load()
    # 学習データの絞り込み１　＠　時価総額 making now
    stock_con.search_isinrange_marketcap(min_value=min_value, max_value=max_value)
    # 学習データの絞り込み２　学習対象の日付
    stock_con.search_is_YMDbefore(ymd=YMDbefore)
    # 学習データの絞り込み３　＠　過去N日
    stock_con.search_isexist_past_Nday(n_day=n_day)
    stock_con.unit_data()
    # networkの作成
    #network = Network_BasicRNN(unit_amount=unit_amount,
    network = Network_BasicRNN_SoftMax(unit_amount=unit_amount,
                               n_hidden=n_hidden,
                               x=stock_con.data_x,
                               y=stock_con.data_y,
                               clf=clf,
                               layer=layer,
                               learning_rate=learning_rate,
                               key=key,
                               config_path=config_path)

    # 学習
    code_ary = stock_con.get_stockcode_ary()
    learner = Learn(stock_con=stock_con,
                    network=network,
                    test_ratio=test_ratio,
                    unit_amount=unit_amount,
                    batch_size=batch_size,
                    result_path=result_path,
                    code_ary=code_ary,
                    verify_model=verify_model)

    stock_con.save_config(path=result_path)
    learner.learn(epochs=epochs)

def main():
    # iniファイルから読むとかいう処理はこっちで後で書く
    print("making now!")

if __name__ == "__main__":
    test()
