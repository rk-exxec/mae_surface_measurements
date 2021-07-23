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

# This Python file uses the following encoding: utf-8

import os
import time
import numpy as np
import logging
from numpy.lib.npyio import save
import userpaths
from datetime import datetime

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter as VidWriter
import cv2
from PySide2 import QtGui
from PySide2.QtWidgets import QCheckBox, QDial, QDialog, QFileDialog, QGroupBox, QInputDialog, QMessageBox
from PySide2.QtCore import QSettings, QSignalBlocker, QTimer, Qt, Signal, Slot
from vimba.frame import BAYER_PIXEL_FORMATS


from droplet import Droplet, RollingAverager
from additional_gui_elements import CameraSettings
from camera import AbstractCamera, TestCamera, HAS_VIMBA
if HAS_VIMBA:
    from camera import VimbaCamera

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ui_form import Ui_main

USE_TEST_IMAGE = False

class CameraControl(QGroupBox):
    """ 
    Widget to control camera preview, camera object and evaluate button inputs
    """
    update_image_signal = Signal(np.ndarray, bool)
    def __init__(self, parent=None):
        super(CameraControl, self).__init__(parent)
        # loading settings
        settings = QSettings()

        # load UI components
        self.ui: Ui_main = self.window().ui
        self._first_show = True # whether form is shown for the first time

        self.update_timer = QTimer()
        self.update_timer.setInterval(500)
        self.update_timer.setSingleShot(False)
        self.update_timer.timeout.connect(self.update_cam_info)
        
        # video path and writer object to record videos
        self.video_dir = settings.value("camera_control/video_dir", ".", str)
        self.recorder: VidWriter = None

        # initialize camera object
        self.cam: AbstractCamera = None # AbstractCamera as interface class for different cameras
        # if vimba software is installed
        if HAS_VIMBA and not USE_TEST_IMAGE:
            self.cam = VimbaCamera()
            logging.info("Using Vimba Camera")
        else:
            self.cam = TestCamera()
            if not USE_TEST_IMAGE: logging.error('No camera found! Fallback to test cam!')
            else: logging.info("Using Test Camera")
        self.update()
        self._oneshot_eval = False
        self._frametime = RollingAverager(100)
        logging.debug("initialized camera control")

    def __del__(self):
        # self.cam.stop_streaming()
        del self.cam

    def showEvent(self, event):
        """
        custom show event

        connects the signals, requests an initial camear image and triggers camera view setup
        """
        if self._first_show:
            self.connect_signals()
            # on first show take snapshot of camera to display
            self.cam.snapshot()
            # prep preview window
            self.ui.camera_prev.prepare()
            # start cam
            self.ui.startCamBtn.click()

    def closeEvent(self, event: QtGui.QCloseEvent):
        """
        custom close event

        stops camera and video recorder
        """
        # close camera stream and recorder object
        if self.recorder: self.recorder.close()
        self.cam.stop_streaming()
        

    def connect_signals(self):
        """ connect all the signals """
        self.cam.new_image_available.connect(self.update_image)
        self.update_image_signal.connect(self.ui.camera_prev.update_image)
        # button signals
        self.ui.startCamBtn.clicked.connect(self.prev_start_pushed)
        self.ui.oneshotEvalBtn.clicked.connect(self.oneshot_eval)
        self.ui.setROIBtn.clicked.connect(self.apply_roi)
        self.ui.resetROIBtn.clicked.connect(self.reset_roi)
        self.ui.syr_mask_chk.stateChanged.connect(self.needle_mask_changed)

    def is_streaming(self) -> bool:
        """ 
        Return whether camera object is aquiring frames 
        
        :returns: True if camera is streaming, otherwise False
        """
        return self.cam.is_running

    @Slot(int)
    def needle_mask_changed(self, state):
        """called when needel mask checkbox is clicked  
        enables or disables masking of syringe

        :param state: check state of checkbox
        :type state: int
        """
        if (state == Qt.Unchecked):
            self.ui.camera_prev.hide_mask()
        else:
            self.ui.camera_prev.show_mask()

    @Slot()
    def prev_start_pushed(self, event):
        """
        Start Button pushed event

        If camera is not streaming:

        - Start video recorder if corresponding checkbox is ticked

        - Start camera streaming

        - Lock ROI buttons and checkboxes

        If camera is streaming:

        - Stop video recorder if it was active

        - Stop camera streaming

        - Unlock buttons
        """
        if self.ui.startCamBtn.text() != 'Stop':
            if self.ui.record_chk.isChecked():
                self.start_video_recorder()
            self.cam.start_streaming()
            logging.info("Started camera stream")
            self.ui.startCamBtn.setText('Stop')
            self.ui.record_chk.setEnabled(False)
            self.ui.frameInfoLbl.setText('Running')
            self.update_timer.start()
        else:
            self.cam.stop_streaming()
            if self.ui.record_chk.isChecked():
                self.stop_video_recorder()
            self.ui.record_chk.setEnabled(True)
            logging.info("Stop camera stream")
            self.ui.startCamBtn.setText('Start')
            self.cam.snapshot()
            self.ui.frameInfoLbl.setText('Stopped')
            self.ui.processingTimeLbl.setText('')
            self.ui.drpltDataLbl.setText(str(self.ui.camera_prev._droplet))
            self.update_timer.stop()

    def oneshot_eval(self):
        self._oneshot_eval = True
        if not self.cam.is_running: self.cam.snapshot()

    def start_video_recorder(self):
        """
        start the `VidWriter` video recorder

        used to record the video from the camera
        """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recorder = VidWriter(filename=self.video_dir + f"/{now}.mp4",
                                    size=self.cam.get_resolution(),
                                    fps=self.cam.get_framerate(),
                                    codec='mpeg4',
                                    preset='ultrafast',
                                    bitrate='5000k')
        #self.recorder.open(self.video_dir + f"/{now}.mp4", 0x21 ,self.cam.get_framerate(),self.cam.get_resolution())
        logging.info(f"Start video recording. File: {self.video_dir}/{now}.mp4 Resolution:{str(self.cam.get_resolution())}@{self.cam.get_framerate()}")

    def stop_video_recorder(self):
        """
        stops the video recorder
        """
        self.recorder.close()
        #self.recorder.release()
        logging.info("Stoped video recording")

    @Slot()
    def apply_roi(self):
        """ Apply the ROI selected by the rubberband rectangle """
        rubberband = self.ui.camera_prev.get_roi() # new roi selection by rubberband
        if rubberband is None: return
        else:
            x,y,w,h = rubberband
        xc,yc,_,_ = self.cam.get_roi() # current roi
        xm,ym,wm,hm = self.ui.camera_prev.get_mask_dim() # current mask selection
        base_y = self.ui.camera_prev.get_baseline_y() # current baseline
        logging.info(f"Applying ROI pos:({x+xc}, {y+yc}), size:({w}, {h})")
        self.cam.set_roi(x,y,w,h)
        self.ui.camera_prev.set_baseline_y(base_y-(y-yc))
        self.ui.camera_prev.set_mask_dim(xm - (x-xc), ym - (x-yc), wm, hm)

    @Slot()
    def reset_roi(self):
        """ Reset the ROI of the camera """
        xc,yc,_,_ = self.cam.get_roi()
        base_y = self.ui.camera_prev.get_baseline_y()
        xm,ym,wm,hm = self.ui.camera_prev.get_mask_dim()
        logging.info(f"Resetting ROI")
        self.cam.reset_roi()
        self.ui.camera_prev.set_baseline_y(base_y + yc)
        self.ui.camera_prev.set_mask_dim(xm + xc, ym + yc, wm, hm)

    @Slot()
    def increase_exposure(self):
        self.cam.set_exposure(self.cam.get_exposure()*2)

    @Slot()
    def decrease_exposure(self):
        self.cam.set_exposure(self.cam.get_exposure()/2)    
        
    @Slot()
    def set_bright(self):
        self.cam.set_exposure(8000)

    @Slot()
    def set_dark(self):
        self.cam.set_exposure(1000)

    def shift_roi(self, direction:str, fast=False):
        x,y,w,h = self.cam.get_roi() 
        amount = 500 if fast else 50
        if direction == "left":
            self.cam.set_roi(x-amount,y,w,h,nested=False)
        elif direction == "right":
            self.cam.set_roi(x+amount,y,w,h,nested=False)
        elif direction == "up":
            self.cam.set_roi(x,y-amount,w,h,nested=False)
        elif direction == "down":
            self.cam.set_roi(x,y+amount,w,h,nested=False)
        else:
            logging.debug(f"Wrong roi shift direction: {direction}")


    @Slot(np.ndarray)
    def update_image(self, cv_img: np.ndarray):
        """ 
        Slot that gets called when a new image is available from the camera 

        handles the signal :attr:`camera.AbstractCamera.new_image_available`
        
        :param cv_img: the image array from the camera
        :type cv_img: np.ndarray, numpy 2D array

        .. seealso:: :py:meth:`camera_preview.CameraPreview.update_image`, :attr:`camera.AbstractCamera.new_image_available`
        """
        # block image signal to prevent overloading
        #blocker = QSignalBlocker(self.cam)
        
        if self.cam.is_running:
            # evaluate droplet if checkbox checked
            eval = self.ui.evalChk.isChecked() or self._oneshot_eval
            # disable droplet avg for oneshot measurements but not for continous
            if self.ui.evalChk.isChecked():
                self.ui.camera_prev._droplet.averaging = True
            elif self._oneshot_eval:
                self.ui.camera_prev._droplet.averaging = False
            self._oneshot_eval = False
            # display current fps
            self.ui.frameInfoLbl.setText('Running | FPS: ' + str(self.cam.get_framerate()))
            # save image frame if recording
            if self.ui.record_chk.isChecked():
                self.recorder.write_frame(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        elif self._oneshot_eval:
            # enable evaluate for one frame (eg snapshots)
            eval = True
            # disable droplet avg for oneshot measurements
            self.ui.camera_prev._droplet.averaging = False
            self._oneshot_eval = False
        else:
            eval = False

        # if ROI size changed, cause update of internal variables for new image dimensions
        if self.cam._image_size_invalid:
            self.ui.camera_prev.invalidate_imagesize()
            self.cam._image_size_invalid = False

        dt = time.time()
        # update preview image    
        self.ui.camera_prev.update_image(cv_img, eval)
        self.frametime = (time.time() - dt) * 1000
        self.ui.processingTimeLbl.setText(f"{self.frametime:.1f} ms")

        # display droplet parameters
        self.ui.drpltDataLbl.setText(str(self.ui.camera_prev._droplet))

        # unblock signals from cam
        #blocker.unblock()

    @Slot()
    def update_cam_info(self):
        try:
            roi = self.cam.get_roi()
            exp = self.cam.get_exposure()
            self.ui.camInfoLbl.setText(f"Exp: {exp:.1f} | ROI: {roi[2]}x{roi[3]} @ ({roi[0]},{roi[1]})")
        except:
            self.ui.camInfoLbl.setText(f"Error fetching data")

    @property
    def frametime(self):
        """ approx volume of droplet  in px^3 """
        return self._frametime.average

    @frametime.setter
    def frametime(self, value):
        self._frametime._put(value)

    @Slot()
    def set_video_path(self):
        """ update the save path for videos """
        settings = QSettings()
        res = QFileDialog.getExistingDirectory(self, "Select default video directory", ".")
        if (res is not None and res != ""):
            self.video_dir = res
            settings.setValue("camera_control/video_dir", res)
            logging.info(f"set default videodirectory to {res}")

    @Slot()
    def calib_size(self):
        """ 
        map pixels to mm 
        
        - use needle mask to closely measure needle
        - call this fcn
        - enter size of needle
        - fcn calculates scale factor and saves it to droplet object
        """
        if self.ui.camera_prev._needle_mask.isHidden():
            QMessageBox.warning(self, "Error setting scale", "The needle mask needs to be active and set to a known width!")
            return
        # get size of needle mask
        rect = self.ui.camera_prev._mask

        # do oneshot eval and extract height from droplet, then calc scale and set in droplet
        res,ok = QInputDialog.getDouble(self,"Size of calib element", "Please enter the size of the test subject in mm:", 0, 0, 100)
        if not ok or res == 0.0:
            return
        
        droplt = Droplet() # singleton
        droplt.set_scale(res / rect[2])
        QMessageBox.information(self, "Success", f"Scale set to {res / rect[2]:.3f} mm/px")
        logging.info(f"set image to real scale to {res / rect[2]}")

    @Slot()
    def remove_size_calib(self):
        drplt = Droplet()
        drplt.set_scale(None)

    @Slot()
    def save_image_dialog(self):
        raw_image = False

        checkBox = QCheckBox("Save raw image")
        checkBox.setChecked(True)
        msgBox = QMessageBox(self)
        msgBox.setWindowTitle("Save Image")
        msgBox.setText("Save the image displayed in the preview or the raw image from the camera")
        msgBox.setIcon(QMessageBox.Question)
        msgBox.addButton(QMessageBox.Ok)
        msgBox.addButton(QMessageBox.Cancel)
        msgBox.setDefaultButton(QMessageBox.Cancel)
        msgBox.setCheckBox(checkBox)

        val = msgBox.exec()
        if val == QMessageBox.Cancel:
            return
        
        raw_image = msgBox.checkBox().isChecked()
        now = datetime.now().strftime("%y-%m-%d_%H-%M")
        save_path,filter = QFileDialog.getSaveFileName(self,"Choose save file", f"{userpaths.get_my_pictures()}/screenshot_{now}.png","Images (*.png *.jpg *.bmp)")
        if save_path is None:
            return
        self.save_image(save_path,raw_image)
        
    def save_image(self, path:str, raw=True):
        qimg = self.ui.camera_prev.grab_image(raw=raw)
        if qimg is None:
            return

        qimg.save(path, quality=100)

    @Slot()
    def camera_settings_dialog(self):
        dlg = CameraSettings(self, self.cam)
        dlg.exec_()

    @Slot()
    def reset_camera(self):
        self.cam.reset()
