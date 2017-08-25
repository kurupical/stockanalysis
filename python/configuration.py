

class Configuration:

  @staticmethod
  def parse_from_file():
    config = ConfigParser()
    config.read(file)
    codes = config['param']['codes']
    codes = str_to_list(str=codes, split_char=",")
    start_money = int(config['param']['start_money'])
    start_date = config['param']['start_date']
    test_term = int(config['param']['test_term'])
