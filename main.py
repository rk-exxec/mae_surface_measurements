#     MAEsure is a program to measure the surface energy of MAEs via contact angle
#     Copyright (C) 2021  Raphael Kriegl

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys

from PySide2 import QtGui


sys.path.append('./src')
import os
import pathlib
import subprocess
import pydevd
import gc
import atexit
import logging
from PySide2.QtGui import QKeyEvent, QKeySequence, QResizeEvent, QPixmap
from PySide2.QtWidgets import QMainWindow, QApplication, QShortcut, QSplashScreen
from PySide2.QtCore import QCoreApplication, QSettings, Qt

from camera_control import CameraControl
from data_control import DataControl
from additional_gui_elements import AboutDialog, CamInfoDialog
from droplet import Droplet

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui : Ui_main = None
        atexit.register(self.cleanup)
        #self._size = self.size()

    def __del__(self):
        del self.ui.camera_prev

    def resizeEvent(self, event: QResizeEvent):
        # prevent resizing of window
        #self.setFixedSize(self._size)
        pass

    def closeEvent(self, event):
        self.ui.camera_ctl.closeEvent(event)
        self.ui.dataControl.closeEvent(event)
        self.ui.idCombo.closeEvent(event)
        return super().closeEvent(event)

    def register_action_events(self):
        # action menu signals
        self.ui.actionVideo_Path.triggered.connect(self.ui.camera_ctl.set_video_path)
        self.ui.actionKalibrate_Size.triggered.connect(self.ui.camera_ctl.calib_size)
        self.ui.actionDelete_Size_Calibration.triggered.connect(self.ui.camera_ctl.remove_size_calib)
        self.ui.actionSave_Image.triggered.connect(self.ui.camera_ctl.save_image_dialog)
        self.ui.actionCameraSettings.triggered.connect(self.ui.camera_ctl.camera_settings_dialog)
        self.ui.actionReset_Camera.triggered.connect(self.ui.camera_ctl.reset_camera)
        # menu reset settings
        self.ui.actionSyringe_Mask.triggered.connect(self.ui.camera_prev._needle_mask.delete_geo)
        self.ui.actionBaseline.triggered.connect(self.ui.camera_prev._baseline.delete_y_level)
        self.ui.actionSample_IDs.triggered.connect(self.ui.idCombo.delete_entries)
        self.ui.actionCamera_scale_factor.triggered.connect(Droplet.delete_scale)
        # info menu 
        self.ui.actionCamera_Info.triggered.connect(lambda: CamInfoDialog(self.window(), self.ui.camera_ctl.cam))
        self.ui.actionAbout_MAEsure.triggered.connect(lambda : AboutDialog(self.window()))
        # shortcuts
        QShortcut(QtGui.QKeySequence("F5"), self, lambda: self.ui.measurementControl.continue_measurement())
        QShortcut(QtGui.QKeySequence("F6"), self, lambda: self.ui.pump_control.infuse())
        QShortcut(QtGui.QKeySequence("F9"), self, self.ui.camera_ctl.set_bright)
        QShortcut(QtGui.QKeySequence("F10"), self, self.ui.camera_ctl.set_dark)
        QShortcut(QtGui.QKeySequence("F12"), self, self.ui.camera_ctl.save_image_dialog)

        QShortcut(QtGui.QKeySequence("PgUp"), self, self.ui.camera_ctl.increase_exposure)
        QShortcut(QtGui.QKeySequence("PgDown"), self, self.ui.camera_ctl.decrease_exposure)

        QShortcut(QtGui.QKeySequence("left"), self, lambda: self.ui.camera_ctl.shift_roi("left"))
        QShortcut(QtGui.QKeySequence("Shift+left"), self, lambda: self.ui.camera_ctl.shift_roi("left",True))
        QShortcut(QtGui.QKeySequence("right"), self, lambda: self.ui.camera_ctl.shift_roi("right"))
        QShortcut(QtGui.QKeySequence("Shift+right"), self, lambda: self.ui.camera_ctl.shift_roi("right",True))
        QShortcut(QtGui.QKeySequence("up"), self, lambda: self.ui.camera_ctl.shift_roi("up"))
        QShortcut(QtGui.QKeySequence("Shift+up"), self, lambda: self.ui.camera_ctl.shift_roi("up",True))
        QShortcut(QtGui.QKeySequence("down"), self, lambda: self.ui.camera_ctl.shift_roi("down"))
        QShortcut(QtGui.QKeySequence("Shift+down"), self, lambda: self.ui.camera_ctl.shift_roi("down",True))

        QShortcut(QtGui.QKeySequence("Backspace"), self, lambda: self.ui.camera_prev.invalidate_droplet())
        

    def cleanup(self):
        del self

def initialize_logger(out_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
     
    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    #formatter = logging.Formatter("%(levelname)s - %(message)s")
    #handler.setFormatter(formatter)
    logger.addHandler(handler)
 
    # create error file handler and set level to error
    handler = logging.FileHandler(os.path.join(out_dir, "error.log"),"w", encoding=None, delay="true")
    handler.setLevel(logging.ERROR)
    #formatter = logging.Formatter("%(levelname)s - %(message)s")
    #handler.setFormatter(formatter)
    logger.addHandler(handler)
 
    # create debug file handler and set level to debug
    handler = logging.FileHandler(os.path.join(out_dir, "all.log"),"w")
    handler.setLevel(logging.INFO)
    #formatter = logging.Formatter("%(levelname)s - %(message)s")
    #handler.setFormatter(formatter)
    logger.addHandler(handler)

class App(QApplication):

    def __init__(self, *args, **kwargs):
        super(App,self).__init__(*args, **kwargs)
        pic = QPixmap('qt_resources/maesure.png')
        splash = QSplashScreen(pic)#, Qt.WindowStaysOnTopHint)
        #splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        splash.setMask(pic.mask())
        splash.show()
        self.window = MainWindow()
        self.ui = Ui_main()
        self.window.ui = self.ui
        self.ui.setupUi(self.window)
        self.window.register_action_events()
        self.window.show()
        
        splash.finish(self.window)

if __name__ == "__main__":
    # compile python qt form.ui into python file
    # uic_path = pathlib.PurePath(os.path.dirname(sys.executable)) / "Scripts"
    # py_file_path = pathlib.PurePath(os.getcwd()) / 'src/ui_form.py'
    # ui_file_path = pathlib.PurePath(os.getcwd()) / 'qt_resources/form.ui'
    # command = f"pyside2-uic -o {py_file_path.as_posix()} {ui_file_path.as_posix()}"
    # print(command)
    # p = subprocess.Popen(command, cwd=uic_path.as_posix())
    # p.wait()
    os.system("pyside2-uic -o src/ui_form.py qt_resources/form.ui")

    from ui_form import Ui_main

    # setup logging
    initialize_logger("./log")

    # pysde2 settings config
    QCoreApplication.setOrganizationName("OTH Regensburg")
    QCoreApplication.setApplicationName("MAEsure")

    # init application
    app = App(sys.argv)
    app.processEvents()

    # execute qt main loop
    sys.exit(app.exec_())
