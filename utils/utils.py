import cv2
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import Qt


class Utils:
    @staticmethod
    def build_model(path):
        """
        Преобразует файл весов .onnx в модель нейронной сети.

        :param path: путь до весов
        :return: модель нейронной сети, построенная на основе весов
        """
        net = cv2.dnn.readNet(path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    @staticmethod
    def convert_frame_to_yolov5_format(frame):
        """
        Преобразует входной кадр в yolov5 формат.

        :param frame: кадр с веб-камеры
        :return: матрица, подходящая для обработки
        """
        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    @staticmethod
    def convert_frame_to_qt_format(frame):
        """
        Преобрузет кадр в qt формат.

        :param frame: входной кадр
        :return: кадр, преобразованный в qt формат
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        result_img = qt_img.scaled(480, 480, Qt.KeepAspectRatio)
        return result_img
