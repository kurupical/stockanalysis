# stock_analysis library
import trade
import trade_algorithm
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
    config = cfp.ConfigParser()
    config.read(file)
    # Predicterの作成
    codes = config['param']['codes']
    start_money = int(['param']['start_money'])
    start_date = ['param']['start_date']
    test_term = int(['param']['test_term'])
    model_path_ary = config['param']['model_path_ary']
    predicter = trade.predicter(model_path_ary)
    # Algorithmの作成
    tradealgo_param_ary = config['param']['tradealgo_param_ary']
    for algo_param in tradealgo_param_ary:
        algos.append(select_algo(algo_param, predicter))

    return codes, start_date, predicter, algos

def select_algo(algo_param, predicter):
    '''
    iniファイルのパラメータを解釈し、アルゴリズムのオブジェクトを返す
    algo_param[0] : アルゴリズムの名前
    algo_param[1:] : アルゴリズムのパラメータ
    '''
    if algo_param[0] == "updown":
        # [1] : n_percent
        return trade_algorithm.UpDown_Npercent(preciter=predicter,
                                               n_percent=algo_param[1])

if __name__ == "__main__":

    dirs = os.listdir(TEST_PATH)
    for dir in dirs:
        # フォルダ名に"_test"が含まれるもの、読み込み対象とする
        if os.path.isdir(TEST_PATH + dir) and re.match(r"(.*)_test", dir) is not None:
            # iniファイルの読み込み
            ini_file = TEST_PATH + dir + "/testpattern.ini"
            codes, start_money, start_date, test_term, predicter, tradealgos = read_ini(ini_file)
            # TradeControllerの設定
            trade_con = trade.TradeController(start_money)
            for code in codes:
                trade = trade.Trade(code=code, tradealgo=tradealgos, predicter=predicter, stock_con=stock_con,date_to=start_date)
                trade_con.add_trade(trade)
            for i in range(test_term):
                trade_con.trade()
                trade_con.forward_1day()
