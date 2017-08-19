from stock import *
from unitrule_stock import *
from unitrule_stockcon import *


def test():
    # とりあえずなんか動かしたい時用
    unitrule_s = UnitRule_Stock_ForwardDay(unit_amount=100, forward_day=1, predict_mode="normal")
    unitrule_sc = UnitRule_Stockcon_Bundle()
    stock_con = StockController(csv_path="../dataset/debug/stock_analysis/",
                                unitrule_stock=unitrule_s,
                                unitrule_stockcon=unitrule_sc,
                                input_items=["終値"],
                                output_items=["終値"])
    stock_con.load()
    # 学習データの絞り込み１　＠　時価総額 making now
    # stock_con.search_isinrange_marketcap(min_value=0, max_vaule=10**11)
    # 学習データの絞り込み２　＠　過去N日
    stock_con.search_isexist_past_Nday(n_day=200)
    x, y = stock_con.unit_data()
    # networkの作成

    # 学習/保存

def main():
    # iniファイルから読むとかいう処理はこっちで後で書く
    print("making now!")

if __name__ == "__main__":
    test()
