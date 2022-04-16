import functools
import sys
from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtWidgets import QMessageBox
from start_and_stop_stream import StreamDetectionController
from net import Net
from available_cameras import Cameras
import detect_on_stream


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        """
        Класс главного окна программы.

        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        uic.loadUi("Gui/detection.ui", self)

        self.setWindowIcon(QtGui.QIcon('Gui/icon.ico'))
        self.setWindowTitle('Detection')

        self.msg = QMessageBox()
        self.msg.setWindowTitle('Warning')
        self.detection = detect_on_stream.Detection()
        self.stream_controller = StreamDetectionController(self.stream_label, self.msg, self.detection)

        self.detection.frame_update.connect(self.stream_controller.image_update_slot)
        self.detection.connection_lost.connect(self.stream_controller.connection_lost_stream_label_update_slot)

        self.choose_weights_btn.clicked.connect(
            functools.partial(Net.get_net, self.msg, self.weights_label)
        )

        self.update_cameras_btn.clicked.connect(
            functools.partial(Cameras.show_available_cameras_on_window, self.camera_combobox)
        )

        self.start_btn.clicked.connect(
            functools.partial(self.stream_controller.start, self.camera_combobox)
        )
        self.stop_btn.clicked.connect(
            functools.partial(self.stream_controller.stop)
        )
        self.exit_btn.clicked.connect(exit_program)


def exit_program():
    """
    Выход из программы.

    :return:
    """
    window.close()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
