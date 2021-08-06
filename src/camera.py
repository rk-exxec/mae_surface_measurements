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

import logging
from threading import Thread, Event
import cv2
import pydevd
from PySide2.QtCore import QObject, QTimer, Signal, Slot
import numpy as np

try:
    from vimba import Vimba, Frame, Camera, LOG_CONFIG_TRACE_FILE_ONLY
    from vimba.frame import FrameStatus
    from vimba.error import VimbaCameraError
    HAS_VIMBA = True
except Exception as ex:
    HAS_VIMBA = False

from typing import List, TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from vimba import Vimba, Frame, Camera, LOG_CONFIG_TRACE_FILE_ONLY
    from vimba.frame import FrameStatus
    from vimba.error import VimbaCameraError

class FrameRateCounter:
    """ Framerate counter with rolling average filter """
    def __init__(self, length=5):
        # lenght of filter
        self.length = length
        self.buffer = [50.0]*length
        self.counter = 0
        self.last_timestamp = 0

    def _rotate(self):
        """
        update internal index of filter
        """
        # increase current index by 1 or loop back to 0
        if self.counter == (self.length - 1):
            self.counter = 0
        else:
            self.counter += 1

    def _put(self, value):
        """
        add value to filter and rotate
        """
        # add value to current line then rotate index
        self.buffer[self.counter] = value
        self._rotate()

    @staticmethod
    def _calc_frametime(timestamp_new, timestamp_old):
        """
        Calculate frametime from camera timestamps (in ns)

        :param timestamp_new: current timestamp from camera (ns)
        :param timestamp_old: previous timestamp from camera (ns)
        :returns: time between timestamps in s
        """
        return (timestamp_new - timestamp_old)*1e-9

    @property
    def average_fps(self) -> float:
        """ the averaged fps """
        return sum(self.buffer) / self.length

    def add_new_timesstamp(self, timestamp):
        """ Add new poi to buffer
        Framtime is calculated from timestamp then added to buffer.
        """
        self._put(1 / self._calc_frametime(timestamp, self.last_timestamp))
        self.last_timestamp = timestamp


class AbstractCamera(QObject):
    """ Interface class for implementing camera objects for MAEsure """
    # signal to emit when a new image is available
    new_image_available = Signal(np.ndarray)
    """ signal that emits when camera has new image available

    .. seealso:: :meth:`camera_control.CameraControl.update_image`
    """
    def __init__(self):
        super(AbstractCamera, self).__init__()
        self._is_running = False
        self._image_size_invalid = True

    @property
    def is_running(self):
        """ 
        Check if camera is running

        :returns: True if camera is running, else False 
        """
        return self._is_running

    def snapshot(self):
        """ record a single image and emit signal """
        raise NotImplementedError

    def start_streaming(self):
        """ start streaming """
        raise NotImplementedError

    def stop_streaming(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def set_roi(self, x, y, w, h):
        """ 
        set the region of interest on the cam

        :param x,y: x,y of ROI in image coordinates
        :param w,h: width and height of ROI
        """
        raise NotImplementedError

    def get_roi(self) -> Tuple[int,int,int,int]:
        """ 
        get the region of interest on the cam

        :returns: x,y,w,h of ROI in image coordinates
        """
        raise NotImplementedError

    def reset_roi(self):
        """ Reset ROI to full size """
        raise NotImplementedError

    def get_framerate(self):
        """ return current FPS of camera"""
        pass #raise NotImplementedError

    def get_resolution(self) -> Tuple[int,int]:
        """ return resolution of current camera capture (width, height) """
        raise NotImplementedError

    def get_exposure(self):
        raise NotImplementedError

    def set_exposure(self, exposure):
        raise NotImplementedError

    def get_exposure_range(self):
        raise NotImplementedError

    def get_exposure_mode(self):
        raise NotImplementedError

    def set_exposure_mode(self, mode):
        raise NotImplementedError


if HAS_VIMBA:
    class VimbaCamera(AbstractCamera):
        """ provides interface to the Allied Vision Camera """
        def __init__(self):
            super(VimbaCamera, self).__init__()
            self._stream_killswitch: Event = None
            self._frame_producer_thread: Thread = None
            self._cam : Camera = None
            self._cam_id: str = ''
            self._vimba: Vimba = Vimba.get_instance()
            self._frc = FrameRateCounter(10)
            #self._vimba.enable_log(LOG_CONFIG_TRACE_FILE_ONLY)
            self._init_camera()
            self._prime_vimba()
            self._setup_camera()

        def __del__(self):
            self.stop_streaming()
            del self._cam
            del self._vimba
        
        def _prime_vimba(self):
            # prime vimba interface, so it doesnt shut down
            # speeds up all camera actions
            self._vimba.__enter__()
            self._cam.__enter__()

        def reset(self):
            self.stop_streaming()
            self._cam.__exit__(None, None, None)
            self._vimba.__exit__(None, None, None)
            self._vimba.__enter__()
            self._cam.__enter__()

        def snapshot(self):
            if self._is_running: return
            with self._vimba:
                with self._cam:
                    frame: Frame = self._cam.get_frame()
                    self.new_image_available.emit(frame.as_opencv_image())

        def stop_streaming(self):
            if self._is_running:
                self._stream_killswitch.set() # set the event the producer is waiting on
                self._frame_producer_thread.join() # wait for the thread to actually be done
                self._is_running = False

        def start_streaming(self):
            self._is_running = True
            self._stream_killswitch = Event() # the event that will be used to stop the streaming
            self._frame_producer_thread = Thread(target=self._frame_producer) # create the thread object to run the frame producer
            self._frame_producer_thread.start() # actually start the thread to execute the method given as target

        def reset_roi(self):
            was_running = self._is_running
            self.stop_streaming()
            with self._vimba:
                with self._cam:
                    self._cam.OffsetX.set(0)
                    self._cam.OffsetY.set(0)
                    _, w_max = self._cam.Width.get_range()
                    _, h_max = self._cam.Height.get_range()
                    self._cam.Width.set(w_max)
                    self._cam.Height.set(h_max)
            self._image_size_invalid = True
            self.snapshot()
            if was_running: self.start_streaming()

        def set_roi(self, x, y, w, h, nested=True):
            was_running = self._is_running
            self.stop_streaming()
            with self._vimba:
                with self._cam:
                    # get current ROI
                    xo,yo,cw,ch = self.get_roi()
                    if nested:
                        # support nested ROI by adding current offset to new one
                        xo = x+xo
                        yo = y+yo
                        # limit width / height to stay within boundaries of new offset and current width/heigth
                        w = min(w, cw - x)
                        h = min(h, ch - y)
                    else:
                        xo = x
                        yo = y

                    # get width/height range and step size
                    w_step = self._cam.Width.get_increment()
                    h_step = self._cam.Height.get_increment()
                    w_min, w_max = self._cam.Width.get_range()
                    h_min, h_max = self._cam.Height.get_range()

                    # clamp values to range and step size
                    w = (w - w_min) // w_step * w_step + w_min
                    h = (h - h_min) // h_step * h_step + h_min
                    w = max(w_min, min(w_max, w))
                    h = max(h_min, min(h_max, h))

                    # set width and height
                    self._cam.Width.set(w)
                    self._cam.Height.set(h)

                    # get offset range and step
                    xo_step = self._cam.OffsetX.get_increment()
                    yo_step = self._cam.OffsetY.get_increment()
                    xo_min, xo_max = self._cam.OffsetX.get_range()
                    yo_min, yo_max = self._cam.OffsetY.get_range()

                    #clamp offset to range and step size
                    xo = (xo - xo_min) // xo_step * xo_step + xo_min
                    yo = (yo - yo_min) // yo_step * yo_step + yo_min
                    xo = max(xo_min, min(xo_max, xo))
                    yo = max(yo_min, min(yo_max, yo))

                    self._cam.OffsetX.set(xo)
                    self._cam.OffsetY.set(yo)

            self._image_size_invalid = True
            self.snapshot()
            if was_running: self.start_streaming()
                
        def get_roi(self):
            with self._vimba:
                with self._cam:
                    w = self._cam.Width.get()
                    h = self._cam.Height.get()
                    x = self._cam.OffsetX.get()
                    y = self._cam.OffsetY.get()
            return x,y,w,h

        def _frame_producer(self):
            with self._vimba:
                with self._cam:
                    try:
                        self._cam.start_streaming(handler=self._frame_handler, buffer_count=10)
                        self._stream_killswitch.wait()
                    finally:
                        self._cam.stop_streaming()

        def _frame_handler(self, cam: Camera, frame: Frame) -> None:
            #pydevd.settrace(suspend=False)
            self._frc.add_new_timesstamp(frame.get_timestamp())
            if frame.get_status() != FrameStatus.Incomplete:
                img = frame.as_opencv_image()
                self.new_image_available.emit(img)
            cam.queue_frame(frame)

        def _init_camera(self):
            with self._vimba:
                cams = self._vimba.get_all_cameras()
                self._cam = cams[0]
                with self._cam:
                    self._cam.AcquisitionStatusSelector.set('AcquisitionActive')
                    if self._cam.AcquisitionStatus.get():
                        self._cam.AcquisitionStop.run()
                        # fetch broken frame
                        self._cam.get_frame()

        def _reset_camera(self):
            with self._cam:
                self._cam.DeviceReset.run()

        def _setup_camera(self):
            #self.reset_camera()
            with self._vimba:
                with self._cam:
                    self._cam.ExposureTime.set(1000.0)
                    self._cam.ReverseY.set(True)
                    self._cam.AcquisitionFrameRateEnable.set(True)
                    self._cam.AcquisitionFrameRate.set(30.0)

        def get_framerate(self):
            try:
                with self._vimba:
                    with self._cam:
                        return round(self._cam.AcquisitionFrameRate.get(),2)
                        #return round(self._frc.average_fps,1)
            except VimbaCameraError as ex:
                return -1

        def get_resolution(self) -> Tuple[int, int]:
            with self._vimba:
                with self._cam:
                    res_x = self._cam.Width.get()
                    res_y = self._cam.Height.get()
                    return (res_x, res_y)
        
        def get_exposure(self):
            with self._vimba:
                with self._cam:
                    return self._cam.ExposureTime.get()

        def set_exposure(self, exposure: float):
            with self._vimba:
                with self._cam:
                    self._cam.ExposureTime.set(exposure)

        def get_exposure_range(self):
            with self._vimba:
                with self._cam:
                    min,max = self._cam.ExposureTime.get_range()
            return min,max

        def get_exposure_mode(self):
            with self._vimba:
                with self._cam:
                    mode = self._cam.ExposureAuto.get()
                    logging.debug(f"Exposure mode read: {mode}")
            return mode

        def set_exposure_mode(self, mode):
            if mode not in ["Off", "Once", "Continuous"]:
                raise ValueError()
            with self._vimba:
                with self._cam:
                    self._cam.ExposureAuto.set(mode)


class TestCamera(AbstractCamera):
    def __init__(self):
        super(TestCamera, self).__init__()
        self._timer = QTimer()
        self._timer.setInterval(16)
        self._timer.setSingleShot(False)
        self._timer.timeout.connect(self._timer_callback)
        self._test_image:np.ndarray = cv2.imread('untitled1.png', cv2.IMREAD_GRAYSCALE)
        self._test_image = np.reshape(self._test_image, self._test_image.shape + (1,) )

    def snapshot(self):
        if not self._is_running:
            self.new_image_available.emit(self._test_image.copy())

    def start_streaming(self):
        self._is_running = True
        self._timer.start()

    def stop_streaming(self):
        self._timer.stop()
        self._is_running = False

    def set_roi(self,x,y,w,h):
        pass

    def reset_roi(self):
        pass

    def get_framerate(self):
        return 1000 / self._timer.interval()

    def get_resolution(self) -> Tuple[int, int]:
        return self._test_image.shape

    @Slot()
    def _timer_callback(self):
        self.new_image_available.emit(self._test_image.copy())

class VideoStream(AbstractCamera):
    def __init__(self):
        super(VideoStream, self).__init__()

    def snapshot(self):
        """ record a single image and emit signal """
        raise NotImplementedError

    def start_streaming(self):
        """ start streaming """
        raise NotImplementedError

    def stop_streaming(self):
        raise NotImplementedError

    def set_roi(self, x, y, w, h):
        """ set the region of interest on the cam
        :param x,y: x,y of ROI in image coordinates
        :param w,h: width and height of ROI
        """
        raise NotImplementedError

    def reset_roi(self):
        """ Reset ROI to full size """
        raise NotImplementedError

    def get_framerate(self):
        """ return current FPS of camera"""
        pass #raise NotImplementedError

    def get_resolution(self) -> Tuple[int,int]:
        """ return resolution of current camera capture (width, height) """
        raise NotImplementedError
