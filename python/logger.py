
class Logger:

    def __init__(self, path, obj):
        self.path = path
        self.obj = obj

    def log(self, str, instance):
        file = open(self.path, 'a')

        # ログに出力する内容の編集
        date =
        class_name =
        str = str

        # ログに出力
