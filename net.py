from PyQt5 import QtWidgets
from utils.utils import Utils


class Net:
    net = None
    net_path = ""

    @staticmethod
    def get_net(msg, weights_label):
        """
        Строит нейронную сеть на основе полученных весов.

        :param msg: сообщение об ошибке
        :param weights_label: лейбл для названия весов
        :return:
        """
        Net.net_path, ok = QtWidgets.QFileDialog.getOpenFileName(directory='D:/GitHub/detectionGood/data/models',
                                                                 filter='*.onnx')
        temp_net = Net.net
        try:
            Net.net = Utils.build_model(Net.net_path)
            Net.__set_net_name_on_window(weights_label)
        except Exception:
            Net.net = temp_net
            msg.setText('Bad weights')
            msg.exec_()

    @staticmethod
    def __set_net_name_on_window(weights_label):
        """
        Вывод название файла весов на экран.

        :param weights_label: лейбл для названия весов
        :return:
        """
        weights_label.setText(Net.net_path.split('/')[-1])
