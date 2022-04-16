from PyQt5.QtGui import QPixmap
from net import Net


class StreamDetectionController:
    def __init__(self, stream_label, msg, detection):
        """
        Конструктор класса.

        :param stream_label: лейбл, в котором будет отображаться поток с веб-камеры
        :param msg: информационное сообщение
        :param detection: класс детекции
        """
        self.stream_label = stream_label
        self.msg = msg
        self.detection = detection

    def image_update_slot(self, image):
        """
        Обновляет кадр внутри лейбла.

        :param image: кадр, который будет отрисовываться внутри лейбла
        :return:
        """
        self.stream_label.setPixmap(QPixmap.fromImage(image))

    def connection_lost_stream_label_update_slot(self, mes):
        """
        Выводит сообщение о потере соединения внутри лейбла.

        :param mes: сообщение о потере соединения
        :return:
        """
        self.stream_label.clear()
        self.stream_label.setText(mes)

    def start(self, combobox):
        """
        Начинает показ внутри лейбла детекции изображения на основе весов и камеры.

        :param combobox: список индексов доступных камер
        :return:
        """
        if Net.net is not None and combobox.count() > 0 and not self.detection.thread_active:
            self.detection.setParams(Net.net, combobox.currentIndex())
            self.detection.start()
        else:
            self.msg.setText('Firstly choose weights and camera(or detection is already running)')
            self.msg.exec_()

    def stop(self):
        """
        Останавливает показ.

        :return:
        """
        if self.detection.thread_active:
            self.detection.stop()
        else:
            self.msg.setText('Firstly press Start button')
            self.msg.exec_()
