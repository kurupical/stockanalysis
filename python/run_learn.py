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
    config = Configuration.parse_from_file()
    unit_amount = int(config['param']['unit_amount'])
    forward_day = int(config['param']['forward_day'])
    predict_mode = config['param']['predict_mode']
    csv_path = config['param']['csv_path']
    stockinfo_path = config['param']['stockinfo_path']
    test_ratio = float(config['param']['test_ratio'])
    batch_size = int(config['param']['batch_size'])
    min_value = int(config['param']['marketcap_min'])
    max_value = int(config['param']['marketcap_max'])
    classify_ratio = float(config['param']['classify_ratio'])
    input_items = str_to_list(str=config['param']['input_items'], split_char=",")
    output_items = str_to_list(str=config['param']['output_items'], split_char=",")
    unitrule_stockcon = config['param']['unitrule_stockcon']
    n_day = int(config['param']['n_day'])
    layer = int(config['param']['layer'])
    n_hidden = int(config['param']['n_hidden'])
    clf = config['param']['clf']
    learning_rate = float(config['param']['learning_rate'])
    key = config['param']['key']
    epochs = int(config['param']['epochs'])
    result_path = "../result/" + datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S") + "/"
    config_path = result_path + "net_config.ini"
    verify_model = config['param']['verify_model']
    YMDbefore = config['param']['YMDbefore']

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
    n_in = len(stock_con.data_x[0,0])
    n_out = len(stock_con.data_y[0])
    network = Network_BasicRNN_SoftMax(unit_amount=unit_amount,
                               n_hidden=n_hidden,
                               n_in=n_in,
                               n_out=n_out,
                               clf=clf,
                               layer=layer,
                               learning_rate=learning_rate,
                               key=key,
                               config_path=config_path,
                               classify_ratio=classify_ratio)

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

def main(config):
    # iniファイルから読むとかいう処理はこっちで後で書く
    # @param
    unit_amount = int(config['param']['unit_amount'])
    forward_day = int(config['param']['forward_day'])
    predict_mode = config['param']['predict_mode']
    csv_path = config['param']['csv_path']
    stockinfo_path = config['param']['stockinfo_path']
    test_ratio = float(config['param']['test_ratio'])
    batch_size = int(config['param']['batch_size'])
    min_value = int(config['param']['marketcap_min'])
    max_value = int(config['param']['marketcap_max'])
    classify_ratio = float(config['param']['classify_ratio'])
    input_items = str_to_list(str=config['param']['input_items'], split_char=",")
    output_items = str_to_list(str=config['param']['output_items'], split_char=",")
    unitrule_stockcon = config['param']['unitrule_stockcon']
    n_day = int(config['param']['n_day'])
    layer = int(config['param']['layer'])
    n_hidden = int(config['param']['n_hidden'])
    clf = config['param']['clf']
    learning_rate = float(config['param']['learning_rate'])
    key = config['param']['key']
    epochs = int(config['param']['epochs'])
    result_path = "../result/" + datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S") + "/"
    config_path = result_path + "net_config.ini"
    verify_model = config['param']['verify_model']
    YMDbefore = config['param']['YMDbefore']

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
    n_in = len(stock_con.data_x[0,0])
    n_out = len(stock_con.data_y[0])
    network = Network_BasicRNN_SoftMax(unit_amount=unit_amount,
                               n_hidden=n_hidden,
                               n_in=n_in,
                               n_out=n_out,
                               clf=clf,
                               layer=layer,
                               learning_rate=learning_rate,
                               key=key,
                               config_path=config_path,
                               classify_ratio=classify_ratio)

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

if __name__ == "__main__":
    #test()
    input_path = "../learn_ini/*.ini"
    files = glob.glob(input_path)
    for file in files:
        config = Configuration.parse_from_file(path=file)
        main(config)
