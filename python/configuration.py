# stockanalysis library
from common import *
# common library
from os import rmdir
from configparser import *
from itertools import product
from glob import glob

class Configuration:

    @staticmethod
    def parse_from_file(path):
        # iniファイルから変数を取り込む
        Configuration.config = ConfigParser()
        Configuration.config.read(path)

    def make_config():
        config = ConfigParser()
        config.read("../learn_ini/ini_maker/base/ini_maker.ini")
        ary_unit_amount         = str_to_list(str=config['param']['unit_amount'], split_char=",")
        ary_forward_day         = str_to_list(str=config['param']['forward_day'], split_char=",")
        ary_predict_mode        = str_to_list(str=config['param']['predict_mode'], split_char=",")
        ary_csv_path            = str_to_list(str=config['param']['csv_path'], split_char=",")
        ary_stockinfo_path      = str_to_list(str=config['param']['stockinfo_path'], split_char=",")
        ary_test_ratio          = str_to_list(str=config['param']['test_ratio'], split_char=",")
        ary_batch_size          = str_to_list(str=config['param']['batch_size'], split_char=",")
        ary_min_value           = str_to_list(str=config['param']['marketcap_min'], split_char=",")
        ary_max_value           = str_to_list(str=config['param']['marketcap_max'], split_char=",")
        ary_classify_ratio      = str_to_list(str=config['param']['classify_ratio'], split_char=",")
        # input_items/output_items は「終値」のみ対応
        ary_input_items         = str_to_list(str=config['param']['input_items'], split_char=",")
        ary_output_items        = str_to_list(str=config['param']['output_items'], split_char=",")
        ary_unitrule_stockcon   = str_to_list(str=config['param']['unitrule_stockcon'], split_char=",")
        ary_n_day               = str_to_list(str=config['param']['n_day'], split_char=",")
        ary_layer               = str_to_list(str=config['param']['layer'], split_char=",")
        ary_n_hidden            = str_to_list(str=config['param']['n_hidden'], split_char=",")
        ary_clf                 = str_to_list(str=config['param']['clf'], split_char=",")
        ary_learning_rate       = str_to_list(str=config['param']['learning_rate'], split_char=",")
        ary_key                 = str_to_list(str=config['param']['key'], split_char=",")
        ary_epochs              = str_to_list(str=config['param']['epochs'], split_char=",")
        ary_verify_model        = str_to_list(str=config['param']['verify_model'], split_char=",")
        ary_YMDbefore           = str_to_list(str=config['param']['YMDbefore'], split_char=",")

        ary_configs = product(
            ary_unit_amount,
            ary_forward_day,
            ary_predict_mode,
            ary_csv_path,
            ary_stockinfo_path,
            ary_test_ratio,
            ary_batch_size,
            ary_min_value,
            ary_max_value,
            ary_classify_ratio,
            ary_input_items,
            ary_output_items,
            ary_unitrule_stockcon,
            ary_n_day,
            ary_layer,
            ary_n_hidden,
            ary_clf,
            ary_learning_rate,
            ary_key,
            ary_epochs,
            ary_verify_model,
            ary_YMDbefore
        )

        # ファイルの削除
        input_path = "../learn_ini/ini_maker/*.ini"
        files = glob(input_path)
        for file in files:
            os.remove(file)

        i = 0
        for configs in ary_configs:
            config = ConfigParser()
            config['param'] = {
                'unit_amount'         : int(configs[0]),
                'forward_day'         : int(configs[1]),
                'predict_mode'        : configs[2],
                'csv_path'            : configs[3],
                'stockinfo_path'      : configs[4],
                'test_ratio'          : float(configs[5]),
                'batch_size'          : int(configs[6]),
                'marketcap_min'       : int(configs[7]),
                'marketcap_max'       : int(configs[8]),
                'classify_ratio'      : float(configs[9]),
                'input_items'         : configs[10],
                'output_items'        : configs[11],
                'unitrule_stockcon'   : configs[12],
                'n_day'               : int(configs[13]),
                'layer'               : int(configs[14]),
                'n_hidden'            : int(configs[15]),
                'clf'                 : configs[16],
                'learning_rate'       : float(configs[17]),
                'key'                 : str(configs[18]),
                'epochs'              : int(configs[19]),
                'verify_model'        : configs[20],
                'YMDbefore'           : configs[21]
            }
            with open("../learn_ini/ini_maker/" + str(i) + ".ini", "w") as configfile:
                config.write(configfile)
            i += 1

if __name__ == "__main__":
    Configuration.make_config()
