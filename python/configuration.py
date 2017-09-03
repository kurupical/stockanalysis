
# common library
from configparser import *

class Configuration:

    @staticmethod
    def parse_from_file(path="../learn_ini/learn.ini"):
        # iniファイルから変数を取り込む
        config = ConfigParser()
        config.read(path)
        return config
