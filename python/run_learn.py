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
    stock_con.unit_data()

def main():
    # iniファイルから読むとかいう処理はこっちで後で書く
    print("making now!")

if __name__ == "__main__":
    test()
