from PyQt5.QtMultimedia import QCameraInfo


class Cameras:
    @staticmethod
    def show_available_cameras_on_window(camera_combobox):
        """
        Вывод в комбо бокс список доступных веб-камер.

        :param camera_combobox: комбобокс, в который помещается список доступных камер
        :return:
        """
        camera_combobox.clear()
        for i in range(len(QCameraInfo.availableCameras())):
            camera_combobox.addItem('Camera #' + str(i + 1))
