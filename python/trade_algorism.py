import trade
import learn

class TradeAlgorism:
# 基底クラス
    def __init__(self, predictor):
        self.predictor = predictor

class up_10percent(TradeAlgorism):
    def __init__(self, predictor):
        super().__init__(predictor)

    def judge(self):
        print("making!")
