

class AssetManager:
    def __init__(self):
        self.assetmng_ary = []

    def add_assetmng(self, assetmng_obj):
        self.assetmng_ary.append(assetmng_obj)


class MustHave_Npercent_Money:
    '''
    取引により現金が総資産のN%を切る場合、購入しない
    '''
    def __init__(self, n_percent):
        self.n_percent = n_percent

    def check(self, trade_con, code):
        trade_obj = trade_con.get_trade_obj(code)
        price_today = trade_obj.chart.get_today_price()
        

class MustnotBuy_Npercent_Per_stock:
    '''
    取引により該当株の資産が総資産のN%を超える場合、購入しない
    '''
    def __init__(self, n_percent):
        self.n_percent = n_percent

    def check(self, trade_con, code):


class MustnotBuy_Nday:
    '''
    同一銘柄が過去N日取引されている場合、購入しない
    '''
    def __init__(self, n_day):
        self.n_day = n_day

    def check(self, trade_con, code):
