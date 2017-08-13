
import edit
import learn

if __name__ == '__main__':
    # edit.run()
    unit = [50,100]
    learning_rate = [0.01, 0.005, 0.001]
    n_hidden = [50, 100, 500, 1000]
    classifier = ["RNN", "LSTM", "GRU"]
    for un in unit:
        for lr in learning_rate:
            for hid in n_hidden:
                for clf in classifier:
                    learn.run(unit=un, \
                                learning_rate=lr, \
                                n_hidden=hid, \
                                epochs=5000, \
                                batch_size=40, \
                                clf=clf)
