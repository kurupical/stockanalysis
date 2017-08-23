from common import *

class AssetManager:
    def __init__(self):
        self.assetmng_ary = []

    def add_assetmng(self, assetmng_obj):
        self.assetmng_ary.append(assetmng_obj)

    def judge(self, trade_con, code, amount):
        for obj in self.assetmng_ary:
            if obj.judge(trade_con, code, amount) is False:
                return False
        return True

class MustHave_Npercent_Money:
    '''
    取引により現金が総資産のN%を切る場合、購入しない
    '''
    def __init__(self, n_percent):
        self.n_percent = n_percent

    def judge(self, trade_con, code, amount):
        chart = trade_con.get_chart(code)
        price_today = chart.get_today_price() * amount
        asset_n_percent = trade_con.total_asset * self.n_percent / 100

        print("debug: total_asset=", trade_con.total_asset)
        print("debug: price_today=", price_today)
        print("debug: asset_n_percent=", asset_n_percent)
        if (trade_con.money - price_today) < asset_n_percent:
            return False
        else:
            return True

class MustnotBuy_Npercent_Per_stock:
    '''
    取引により該当株の資産が総資産のN%を超える場合、購入しない
    '''
    def __init__(self, n_percent):
        self.n_percent = n_percent

    def judge(self, trade_con, code, amount):
        chart = trade_con.get_chart(code)
        price_today = chart.get_today_price() * amount
        asset_n_percent = trade_con.total_asset * self.n_percent / 100

        # 保有分＋今回購入した場合の金額を計算
        if len(trade_con.holdstock) > 0:
            hold = trade_con.holdstock[trade_con.holdstock["code"] == code]
            sum_of_price = (hold["price"].values * hold["amount"].values).sum()
        else:
            sum_of_price = price_today

        print(sum_of_price)
        print(asset_n_percent)
        if sum_of_price > asset_n_percent:
            return False
        else:
            return True

class MustnotBuy_LastBuyAfterNday:
    '''
    同一銘柄が過去N日取引されている場合、購入しない
    '''
    def __init__(self, n_day):
        self.n_day = n_day

    def judge(self, trade_con, code, amount):
        if len(trade_con.holdstock) > 0:
            chart = trade_con.get_chart(code)
            hold = trade_con.holdstock[trade_con.holdstock["code"] == code]
            max_date_of_hold = hold["date"].max()
            today_date = chart.get_today_date()
            if max_date_of_hold + self.n_day > today_date:
                return False
            else:
                return True
        else:
            return True
