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


import math
from evaluate_droplet import Droplet
import logging
import os

import numpy as np
import time
from qthread_worker import CallbackWorker

from PySide2 import QtGui
from PySide2.QtWidgets import QApplication, QGroupBox, QMessageBox
from PySide2.QtCore import QMutex, QObject, QTimer, QWaitCondition, Signal, Slot, Qt

from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from ui_form import Ui_main

class IntervalTimer(QObject):
    """Creates a timer for each entry in an array of time intervals; 
    every timer will call the target function, last timer calls done_callback; 
    allows for precise time control  

    :param parent: containing Qt object
    :param intervals: list of floats representing times in seconds
    :param target: target function to execute after each interval
    :param done_callback: callback function after all timers are finished
    """
    def __init__(self, parent, intervals:List[float], target, done_callback):
        super(IntervalTimer,self).__init__(parent=parent)
        self.target = target
        self.callback = done_callback
        self.intervals = intervals
        self.timers: List[QTimer] = []
        self.set_intervals(intervals)

    def set_intervals(self, intervals):
        self.timers.clear()
        self.intervals = intervals
        for val in intervals:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(val*1000)
            timer.setTimerType(Qt.PreciseTimer)
            timer.timeout.connect(self.target)
            if (val == intervals[-1]): timer.timeout.connect(self.callback)
            self.timers.append(timer)

    def start(self):
        for t in self.timers:
            t.start()

    def stop(self):
        for t in self.timers:
            t.stop()     


class MeasurementControl(QGroupBox):
    """
    class that provides a groupbox with UI to control the measurement process
    """
    new_datapoint_signal = Signal(Droplet, int)
    save_data_signal = Signal()
    start_measurement_signal = Signal(bool)
    save_image_signal = Signal(int)
    def __init__(self, parent=None) -> None:
        super(MeasurementControl, self).__init__(parent)
        self.ui: Ui_main = self.window().ui
        self._time_interval = []
        self._magnet_interval = []
        self._cur_magnet_int_idx = 0
        self._method = 0 # sessile
        self._cycles = 1
        self._cycle = 0
        self.aborted = False
        self.stopped = True
        self.timer: IntervalTimer = None
        self.thread: CallbackWorker = None
        self._first_show = True
        self._meas_aborted = False
        self._mutex = QMutex()
        self._wait_condition = QWaitCondition()

    def showEvent(self, event):
        if self._first_show:
            self.connect_signals()
            self.ui.waitForUserLbl.setVisible(False)
            self._first_show = False

    def connect_signals(self):
        self.ui.startMeasBtn.clicked.connect(self.start_stop_btn_pushed)
        self.ui.cancelMeasBtn.clicked.connect(self.abort_measurement)
        self.ui.avgModeCombo.currentIndexChanged.connect(self.change_avg_mode)
        self.new_datapoint_signal.connect(self.ui.dataControl.new_data_point)
        self.save_image_signal.connect(self.ui.dataControl.save_image)
        self.save_data_signal.connect(self.ui.dataControl.save_data)
        self.ui.continueButton.clicked.connect(self.continue_measurement)
        

    ### gui control fcns ###

    @Slot()
    def start_stop_btn_pushed(self):
        """ 
        Starts or stops the measurement and updates the UI
        """
        if self.ui.startMeasBtn.text() == "Start":
            QApplication.processEvents()
            self.start_measurement()
        else:
            self.stop_measurement()
            

    def start_measurement(self):
        """
        checks if measurement is not already running and camera is functional. 
        If conditions met will start measurement thread
        """
        logging.info("Meas_Ctl: Starting measurement")
        # save id field if not already done
        self.ui.idCombo.save_dropdown_entry()

        if not self.ui.camera_ctl.is_streaming():
            QMessageBox.information(self, 'MAEsure Information',' Camera is not running!\nPlease start camera first!', QMessageBox.Ok)
            logging.info('Meas_Ctl: Camera not running')
            return
        if not self.stopped:
            QMessageBox.information(self, 'MAEsure Information',' Cannot start measurement while already running!', QMessageBox.Ok)
            logging.info('Meas_Ctl: Measurement already running')
            return
        if self.ui.magnetControl.needs_reference():
            QMessageBox.information(self, 'MAEsure Information',' Cannot start measurement without referencing motor!', QMessageBox.Ok)
            logging.info('Meas_Ctl: Motor not referenced!')
            return
        try:
            self.read_intervals()
            self.ui.dataControl.init_data()
        except Exception as ex:
            QMessageBox.warning(self, 'MAEsure Error', f'An error occured:\n{str(ex)}', QMessageBox.Ok)
            logging.exception("Meas_Ctl: error", exc_info=ex)
            return

        # prepare live plot for new data
        self.start_measurement_signal.emit(self.ui.plotHoldChk.isChecked())
        # init local vars
        self.stopped = False
        self.aborted = False
        self._cur_magnet_int_idx = 0
        self._cycles = self.ui.numRepeat.value()
        self._cycle = 0
        self.ui.startMeasBtn.setText("Stop")

        self.measure_start()


    @Slot()
    def stop_measurement(self):
        """
        Stops the measurement gracefully, still writing the data.
        """
        logging.info("Meas_Ctl: stopping measurement")
        self.aborted = False
        self.stopped = True
        self.save_data_signal.emit()
        if self.timer: self.timer.stop()
        self._wait_condition.wakeAll()
        # if self.thread: self.thread.terminate()

        self.ui.startMeasBtn.setText("Start")
        self.ui.continueButton.setEnabled(False)
        self.ui.waitForUserLbl.setVisible(False)

    @Slot()
    def abort_measurement(self):
        """
        stops the measurement without saving data
        """
        logging.info("Meas_Ctl: aborting measurement")
        self.aborted = True
        self.stopped = True
        if self.timer: self.timer.stop()
        self._wait_condition.wakeAll()
        # if self.thread: self.thread.terminate()

        self.ui.startMeasBtn.setText("Start")
        self.ui.continueButton.setEnabled(False)
        self.ui.waitForUserLbl.setVisible(False)

    @Slot()
    def continue_measurement(self):
        """ continue measurement from waiting state """
        self.ui.continueButton.setEnabled(False)
        self.ui.waitForUserLbl.setVisible(False)
        self._wait_condition.wakeAll()
    
    ### measurement control functions ###

    def measure_start(self):
        """ start measurement by driving to initial magnet position if supplied 
        order of running:
        -->
        start_mag_step
        (continue_measurement)
        do_mag_step
        mag_step_done
        -->>
        check_continue
        (wait_continue)
        start_time_sweep
        time_sweep_done
        <<-- repeat, repetitions per mag step
        <-- repeat, every mag step
        sweep_done
        """
        if self.stopped or self.aborted:
            return
        if self.ui.sweepMagChk.isChecked():
            self.start_mag_step()
        else:
            self.start_time_sweep()
        # else:
        #     logging.error("sweep not started: nothing selected!")
        #     return

    def start_mag_step(self):
        """ drive motor to next selected magnet value
         
        """
        if self.stopped or self.aborted:
            return
        self.do_mag_step()
        
    def do_mag_step(self):
        """ drive magnet to next interval pos and wait for movement to finish """
        self.thread = CallbackWorker(target=self.ui.magnetControl.move_pos, pos=self._magnet_interval[self._cur_magnet_int_idx], unit=self.ui.magMeasUnitCombo.currentText(), blocking=True, slotOnFinished=self.mag_step_done)
        self.thread.start()

    @Slot()
    def mag_step_done(self):
        """ one magnet step finished, start new time sweep or cycle """
        if self.stopped or self.aborted:
            return
        self._cur_magnet_int_idx += 1   # next magnet interval
        self._cycle = 0
        self.check_continue(self.start_time_sweep)
        #self.start_time_sweep()

    def check_continue(self, next_action=None):
        """check if we need to wait for user input

        if yes, start a wait thread with the next step as callback
        if no, start the next action immediately

        :param next_action: next action when continuing, defaults to None
        :type next_action: [type], optional
        """
        if self.ui.waitAfterMagChck.isChecked():
            # if wait checkbox is acitve, unlock the contiune button, the button then continues the mag step process
            self.ui.continueButton.setEnabled(True)
            self.ui.waitForUserLbl.setVisible(True)
            self.thread = CallbackWorker(self.wait_continue, slotOnFinished=next_action)
            self.thread.start()
        else:
            next_action()

    def wait_continue(self):
        """wait for continue button pressed
        """
        self._mutex.lock()
        self._wait_condition.wait(self._mutex)
        self._mutex.unlock()

    @Slot()
    def start_time_sweep(self):
        """ starts the sweep by checking if time is to be sweeped , if not it will skip"""
        if self.stopped or self.aborted:
            return

        # try to save droplet image if option selected
        self.save_image_signal.emit(self._cycle)
        if self.ui.sweepTimeChk.isChecked():
            self.ui.dataControl.invalidate_time()
            self.timer = IntervalTimer(self, self._time_interval, self.collect_data, self.time_sweep_done)
            self.timer.start()
        else:
            self.collect_data()
            self.time_sweep_done()

    @Slot()
    def time_sweep_done(self):
        """ after timer has finished or was skipped, moves to next mag step if desired else next cycle """
        if self.stopped or self.aborted:
            return
        if (self._cycle < self._cycles - 1):
            # restart time sweep if repetitions are requested
            self._cycle += 1
            self.check_continue(self.start_time_sweep)
        elif self.ui.sweepMagChk.isChecked() and self._cur_magnet_int_idx < len(self._magnet_interval):
            self.start_mag_step()
        else:
            self.sweep_done()

    @Slot()
    def sweep_done(self):
        """ when timer and magnet sweep has finished """

        self.stop_measurement()
        QMessageBox.information(self, 'MAEsure', 'Measurement finished!', QMessageBox.Ok)
            
    #@Slot()
    def collect_data(self):
        """grab snapshot of data and save to table
        """
        drplt = self.ui.camera_prev._droplet
        logging.debug(f"gathered new datapoint: {drplt.angle_r},{self._cycle}")
        self.new_datapoint_signal.emit(drplt, self._cycle)

    ### utility functions ###
    @Slot(int)
    def change_avg_mode(self, index):
        drplt = Droplet()
        drplt.change_filter_mode(index)

    def read_intervals(self):
        """ Try to read the time and magnet intervals """
        try:
            if self.ui.sweepTimeChk:
                self._time_interval = self.parse_intervals(self.ui.timeInt.text())
                logging.info(f"measurement time interval: {self.ui.timeInt.text()}")
            else:
                self._time_interval = []
        except ValueError as ve:
            QMessageBox.critical(self, 'MAEsure Error!', 'No time values specified! Aborting!', QMessageBox.Ok)
            logging.error('Time interval error: ' + str(ve))

        try:
            if self.ui.sweepMagChk:
                self._magnet_interval = self.parse_intervals(self.ui.magInt.text())
                logging.info(f"measurement magnet interval: {self.ui.magInt.text()}")
            else:
                self._magnet_interval = []
        except ValueError as ve:
            QMessageBox.critical(self, 'MAEsure Error!', 'No magnet values specified! Aborting!', QMessageBox.Ok)
            logging.error('Magnet inteval error: ' + str(ve))
        #self._repeat_after = self.ui.repWhenCombo.currentIndex()

    def parse_intervals(self, expr:str):
        """ Parses string of values to float list

        :param expr: the interval expression

        **expr** can look like '2', '1.0,2.3,3.8', '1.1,3:6' or '1.7,4:6:5,-2.8'  
        - with 'x:y:z' the values will be passed to `numpy.linspace(x, y, num=z, endpoint=True)`, optional z = 10
        - with 'x*y*z' the values will be passed to `numpy.logspace(x, y, num=z, endpoint=True)`, z is optional, def = 10
        *Values won't be sorted*
        """
        expr = expr.strip()
        if expr == '': raise ValueError('Expression empty!')
        calcd_range: List[float] = []
        for expr_part in expr.split(','):
            if ':' in expr_part:
                range_vals = expr_part.split(':')
                range_vals[0] = float(range_vals[0])
                range_vals[1] = float(range_vals[1])
                if len(range_vals) == 2:
                    # results in point in point per second
                    range_vals.append(int(abs(range_vals[1]-range_vals[0])))
                elif len(range_vals) == 3:
                    range_vals[2] = int(range_vals[2])
                calcd_range += list(np.linspace(range_vals[0], range_vals[1], num=range_vals[2], endpoint=True))
            elif '*' in expr_part:
                range_vals = expr_part.split('*')
                range_vals[0] = float(range_vals[0])
                range_vals[1] = float(range_vals[1])
                if len(range_vals) == 2:
                    # try use point every decade
                    range_vals.append(int(math.log(abs(range_vals[1]-range_vals[0]))))
                elif len(range_vals) == 3:
                    range_vals[2] = int(range_vals[2])
                calcd_range += list(np.logspace(range_vals[0], range_vals[1], num=range_vals[2], endpoint=True))
            else:
                calcd_range.append(float(expr_part))
        return calcd_range
