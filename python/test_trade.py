# stock_analysis library
import trade
import trade_algorithm
import asset_manager
import learn
import common
# common library
import configparser as cfp
import glob
import os
import re

# constant
TEST_PATH = "../test/trade/"

def read_ini(file):
    '''
    テスト内容が書かれたiniファイルを読み込み、iniファイルの内容に対応する
    predicterオブジェクトと売買アルゴリズム（配列）を返す
    '''
    algos = []
    assetmng = asset_manager.AssetManager()
    config = cfp.ConfigParser()
    config.read(file)
    # Predicterの作成
    codes = config['param']['codes']
    codes = common.str_to_list(str=codes, split_char="\n")
    start_money = int(config['param']['start_money'])
    start_date = config['param']['start_date']
    test_term = int(config['param']['test_term'])
    model_path_ary = config['param']['model_path_ary']
    model_path_ary = common.str_to_list(str=model_path_ary, split_char="\n")
    predicter = trade.Predicter(model_path_ary)
    # Algorithmの作成
    tradealgo_param_ary = config['param']['tradealgo_param_ary']
    tradealgo_param_ary = common.str_to_list(str=tradealgo_param_ary, split_char="\n")
    for algo_param in tradealgo_param_ary:
        algo_param = common.str_to_list(str=algo_param, split_char=",")
        algos.append(select_algo(algo_param, predicter))

    # AssetManagerの作成
    assetmanager_param_ary = config['param']['assetmanager_param_ary']
    assetmanager_param_ary = common.str_to_list(str=assetmanager_param_ary, split_char="\n")
    for assetmng_param in assetmanager_param_ary:
        assetmng_param = common.str_to_list(str=assetmng_param, split_char=",")
        assetmng.add_assetmng(select_assetmng(assetmng_param))
    return codes, start_money, start_date, test_term, predicter, algos, assetmng

def select_algo(algo_param, predicter):
    '''
    iniファイルのパラメータを解釈し、アルゴリズムのオブジェクトを返す
    algo_param[0] : アルゴリズムの名前
    algo_param[1:] : アルゴリズムのパラメータ
    '''
    if algo_param[0] == "updown":
        # [1] : n_percent
        return trade_algorithm.UpDown_Npercent(predicter=predicter,
                                               n_percent=int(algo_param[1]))
def select_assetmng(assetmng_param):
    '''
    iniファイルのパラメータを解釈し、資産管理ルールのオブジェクトを返す
    algo_param[0] : アルゴリズムの名前
    algo_param[1:] : アルゴリズムのパラメータ
    '''
    if assetmng_param[0] == "MustHave_Npercent_Money":
        return asset_manager.MustHave_Npercent_Money(n_percent=int(assetmng_param[1]))


if __name__ == "__main__":

    dirs = os.listdir(TEST_PATH)
    for dir in dirs:
        # フォルダ名に"_test"が含まれるもの、読み込み対象とする
        if os.path.isdir(TEST_PATH + dir) and re.match(r"(.*)_test", dir) is not None:
            # iniファイルの読み込み
            ini_file = TEST_PATH + dir + "/testpattern.ini"
            codes, start_money, start_date, test_term, predicter, tradealgos, assetmng = read_ini(ini_file)
            # Controllerの設定
            trade_con = trade.TradeController(start_money, assetmng)
            stock_con = learn.StockController()
            stock_con.load()
            for code in codes:
                trade_obj = trade.Trade(code=int(code), tradealgo=tradealgos, predicter=predicter, stock_con=stock_con,date_to=start_date)
                trade_con.add_trade(trade_obj)
            for i in range(test_term):
                trade_con.trade()
                trade_con.forward_1day()
