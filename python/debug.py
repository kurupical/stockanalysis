import learn
import verify

STEP = 100
CSV_PATH = "../dataset/debug/stock_analysis/"

for dim in [2]:
    for i in range(3):
        verify.random_ndim_func(dim=dim, name=CSV_PATH+"debug.csv")
        stock_con = learn.StockController()
        for layer in [1]:
            learn.run(unit=100, \
                    learning_rate=0.01, \
                    n_hidden=50, \
                    epochs=10000, \
                    clf="LSTM", \
                    batch_size=50, \
                    layer=layer, \
                    stock_con=stock_con, \
                    name="dim"+str(dim)+"No_"+str(i))
