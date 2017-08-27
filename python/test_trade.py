# stock_analysis library
from trade import *
from trade_algorithm import *
from asset_manager import *
from common import *
from stock import *
from unitrule_stock import *
from unitrule_stockcon import *
from configuration import *
# common library
from configparser import *
import glob
import os
import re
from predicter import *

# constant
TEST_PATH = "../test/trade/"

def read_testpattern_ini(file):
    '''
    テスト内容が書かれたiniファイルを読み込み、iniファイルの内容に対応する
    predicterオブジェクトと売買アルゴリズム（配列）を返す
    '''
    algos = []
    assetmng = AssetManager()
    config = ConfigParser()
    config.read(file)
    codes = config['param']['codes']
    codes = str_to_list(str=codes, split_char=",")
    start_money = int(config['param']['start_money'])
    start_date = config['param']['start_date']
    test_term = int(config['param']['test_term'])
    # Predicterの作成
    model_path_ary = config['param']['model_path_ary']
    model_path_ary = str_to_list(str=model_path_ary, split_char="\n")
    network_ary = Network.read_network(path_ary=model_path_ary)
    predicter = Predicter.generate_predicter(predicter_model="Predicter_Nto1Predict_MaxMin", network_ary=network_ary)
    # Algorithmの作成
    tradealgo_param_ary = config['param']['tradealgo_param_ary']
    tradealgo_param_ary = str_to_list(str=tradealgo_param_ary, split_char="\n")
    for algo_param in tradealgo_param_ary:
        algo_param = str_to_list(str=algo_param, split_char=",")
        algos.append(select_algo(algo_param, predicter))

    # AssetManagerの作成
    assetmanager_param_ary = config['param']['assetmanager_param_ary']
    assetmanager_param_ary = str_to_list(str=assetmanager_param_ary, split_char="\n")
    for assetmng_param in assetmanager_param_ary:
        assetmng_param = str_to_list(str=assetmng_param, split_char=",")
        assetmng.add_assetmng(select_assetmng(assetmng_param))
    return codes, start_money, start_date, test_term, predicter, algos, assetmng

def select_algo(param, predicter):
    '''
    iniファイルのパラメータを解釈し、アルゴリズムのオブジェクトを返す
    param[0] : アルゴリズムの名前
    param[1:] : アルゴリズムのパラメータ
    '''
    if param[0] == "updown":
        # [1] : n_percent
        return UpDown_Npercent(predicter=predicter, n_percent=int(param[1]))

def select_assetmng(param):
    '''
    iniファイルのパラメータを解釈し、資産管理ルールのオブジェクトを返す
    param[0] : アルゴリズムの名前
    param[1:] : アルゴリズムのパラメータ
    '''
    if param[0] == "MustHave_Npercent_Money":
        return MustHave_Npercent_Money(n_percent=int(param[1]))

    if param[0] == "MustnotBuy_Npercent_Per_stock":
        return MustnotBuy_Npercent_Per_stock(n_percent=int(param[1]))

    if param[0] == "MustnotBuy_LastBuyAfterNday":
        return MustnotBuy_LastBuyAfterNday(n_day=int(param[1]))

def read_stockcon_ini(file):
    # StockControllerに必要なパラメータの設定
    # csv_path
    config = ConfigParser()
    config.read(file)
    csv_path = config['param']['csv_path']
    # UnitRule_Stock
    unitrule_stock_param = config['param']['unitrule_stock']
    unitrule_stock = select_unitrule_stock(unitrule_stock_param)
    # UnitRule_Stockcon
    unitrule_stockcon_param = config['param']['unitrule_stockcon']
    unitrule_stockcon = select_unitrule_stockcon(unitrule_stockcon_param)
    # StockInfo
    stock_info_path = config['param']['stock_info_path']
    stock_info = StockInfo(path=stock_info_path)
    # input_items
    input_items = str_to_list(str=config['param']['input_items'], split_char=",")
    # output_items
    output_items = str_to_list(str=config['param']['output_items'], split_char=",")

    # StockControllerの作成
    stock_con = StockController(csv_path=csv_path,
                                unitrule_stock=unitrule_stock,
                                unitrule_stockcon=unitrule_stockcon,
                                stock_info=stock_info,
                                input_items=input_items,
                                output_items=output_items)

    return stock_con

def select_unitrule_stock(param):
    if param[0] == "UnitRule_Stock_ForwardDay":
        return UnitRule_Stock_ForwardDay(unit_amount=param[1],
                                         forward_day=param[2],
                                         predict_mode=param[3])

def select_unitrule_stockcon(param):
    if param[0] == "Unitrule_Stockcon_Bundle":
        return UnitRule_Stockcon_Bundle()

if __name__ == "__main__":

    dirs = os.listdir(TEST_PATH)
    for dir in dirs:
        # フォルダ名に"_test"が含まれるもの、読み込み対象とする
        if os.path.isdir(TEST_PATH + dir) and re.match(r"(.*)_test", dir) is not None:
            # ログのパスの設定
            Configuration.log_path = TEST_PATH + dir + "/result_log.log"

            ini_file = TEST_PATH + dir + "/testpattern.ini"
            # この処理はConfigurationにいれる。あとでね
            codes, start_money, start_date, test_term, predicter, tradealgos, assetmng = read_testpattern_ini(ini_file)
            # Controllerの設定
            stock_con = read_stockcon_ini(TEST_PATH + dir + "/stock_con/stock_con.ini")
            stock_con.load()
            charts = []
            for code in codes:
                charts.append(Chart(code=int(code), stock_con=stock_con, date_to=start_date))
            trade_con = TradeController(start_money, assetmng, charts)

            for code in codes:
                trade_obj = Trade(code=int(code), tradealgo=tradealgos, predicter=predicter, stock_con=stock_con, date_to=start_date)
                trade_con.add_trade(trade_obj)
            for i in range(test_term):
                trade_con.trade()
                trade_con.forward_1day()
