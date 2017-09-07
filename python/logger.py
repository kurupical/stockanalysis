# stock_analysis library

# common library
import datetime

class Logger:

    def __init__(self, path, obj=None):
        self.path = path
        self.obj = obj

    def log(self, log):
        file = open(self.path, 'a')

        now = datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        if self.obj is not None:
            obj_name = self.obj.__class__.__name__
        else:
            obj_name = ""
        # ログ出力
        file = open(self.path, 'a')
        file.write(str(now) +  "," + str(obj_name) + "," + log + "\n")
        file.close()
