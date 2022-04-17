import cv2
import time
from PyQt5.QtWidgets import QMessageBox
from detect_and_label_on_frame import FrameDetection
from utils.utils import Utils
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage


class Detection(QThread):
    frame_update = pyqtSignal(QImage)
    connection_lost = pyqtSignal(str)

    def __init__(self):
        QThread.__init__(self)
        self.camera_index = None
        self.thread_active = None
        self.net = None
        self.error_msg = QMessageBox()
        self.error_msg.setWindowTitle('Warning')
        self.error_msg.setText('Incompatible weights and classes list')

    def set_params(self, net, camera_index):
        """
        Обновление параметров детекции.

        :param net: текущая нейронная сеть
        :param camera_index: индекс камеры, с которой программа будет получать изображение
        :return:
        """
        self.net = net
        self.camera_index = camera_index

    def run(self):
        """
        Запуск детекции и получения кадра с веб-камеры.

        :return:
        """
        self.thread_active = True
        capture = cv2.VideoCapture(self.camera_index)

        while self.thread_active:
            ret, frame = capture.read()
            if ret:
                detected_img = FrameDetection.detect_and_label(frame, self.net)
                if detected_img is None:
                    self.error_msg.exec_()
                    self.stop()
                else:
                    qt_img = Utils.convert_frame_to_qt_format(detected_img)
                    self.frame_update.emit(qt_img)
            else:
                self.connection_lost.emit('Lost connection, trying to update . . .')
                capture = cv2.VideoCapture(self.camera_index)
                time.sleep(1)

        self.connection_lost.emit('Choose weights, camera and press Start button')

    def stop(self):
        """
        Остановка детекции.

        :return:
        """
        self.thread_active = False
        self.quit()
