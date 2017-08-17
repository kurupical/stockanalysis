

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
        trade_obj = trade_con.get_trade_obj(code)
        price_today = trade_obj.chart.get_today_price() * amount
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
        print("making now!")

class MustnotBuy_Nday:
    '''
    同一銘柄が過去N日取引されている場合、購入しない
    '''
    def __init__(self, n_day):
        self.n_day = n_day

    def judge(self, trade_con, code, amount):
        print("making now!")
