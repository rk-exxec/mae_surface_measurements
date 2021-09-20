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

from math import degrees
from PySide2.QtCore import QSettings
import logging
import inspect

from numpy.lib.function_base import angle

class Singleton(object):
    """ singleton base class """
    _instance = None
    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
            class_._instance._initialized = False
        return class_._instance

# TODO make length always be a fixed fraction of a second and change with framerate

class Droplet(Singleton):
    """
    provides a singleton storage structure for droplet information and conversion functions  

    saves droplet scale with QSettings

    Has the following attributes:

    - **is_valid**: whether contained data is valid
    - **gof**: the goodness of the droplet fit
    - **_angle_l**, **_angle_r**: the unfiltered left and right tangent angles
    - **_angle_l_avg**, **_angle_r_avg**: the rolling averager filter object for the angles
    - **center**: center point of fitted ellipse (x,y)
    - **maj**: length of major ellipse axis
    - **min**: lenght of minor ellipse axis
    - **phi**: tilt of ellipse in rad
    - **tilt_deg**: tilt of ellipse in deg
    - **foc_pt1**, foc_pt2**: focal points of ellipse (x,y)
    - **tan_l_m**, **tan_r_m**: slope of left and right tangent
    - **int_l**, **int_r**: left and right intersections of ellipse with baseline
    - **line_l**, **line_r**: left and right tangent as 4-Tuple (x1,y1,x2,y2)
    - **base_diam**: diameter of the contact surface of droplet
    - **_area**: unfiltered area of droplet silouette
    - **_area_avg**: rolling average filter for area
    - **_height**: unfiltered droplet height in px
    - **_height_avg**: rolling average filter for height
    - **scale_px_to_mm**: scale to convert between px and mm, is loaded from storage on startup
    """
    def __init__(self):
        self.is_valid       : bool                  = False
        if self._initialized: return
        self._initialized = True
        settings                                    = QSettings()
        self.averaging      : bool                  = True
        self.r2             : float                 = 0
        self._angle_l       : float                 = 0.0
        self._angle_l_avg                           = RollingAverager()
        self._angle_r       : float                 = 0.0
        self._angle_r_avg                           = RollingAverager()
        self.center         : tuple[int,int]        = (0,0)
        self.ellipse                                = None
        self.divider        : int                   = 0
        self.tan_l_m        : int                   = 0
        self.int_l          : tuple[int,int]        = (0,0)
        self.line_l         : tuple[int,int,int,int] = (0,0,0,0)
        self.tan_r_m        : int                   = 0
        self.int_r          : tuple[int,int]        = (0,0)
        self.line_r         : tuple[int,int,int,int] = (0,0,0,0)
        self.base_diam      : int                   = 0
        self._volume        : float                 = 0.0
        self._volume_avg                            = RollingAverager()
        self._height        : float                 = 0.0
        self._height_avg                            = RollingAverager()
        var_scale = settings.value("droplet/scale_px_to_mm", 0.0)
        self.scale_px_to_mm : float                 = float(var_scale) if var_scale else None # try to load from persistent storage
        self.error          :str                    = ""

    def __str__(self) -> str:
        if self.is_valid:
            # if scalefactor is present, display in metric, else in pixles
            if self.scale_px_to_mm is None or self.scale_px_to_mm <= 0:
                ret = inspect.cleandoc(f'''
                    Angle Left:
                    {round(self.angle_l,1):.1f}°
                    Angle Right:
                    {round(self.angle_r,1):.1f}°
                    Surface Diam:
                    {round(self.base_diam):.2f} px
                    Volume:
                    {round(self.volume,2):.2f} px3
                    Height:
                    {round(self.height,2):.2f} px
                    R²:
                    {round(self.r2,5)}
                    '''
                )
            else:
                ret =  inspect.cleandoc(f'''
                    Angle Left:
                    {round(self.angle_l,1):.1f}°
                    Angle Right:
                    {round(self.angle_r,1):.1f}°
                    Surface Diam:
                    {round(self.base_diam_mm,2):.2f} mm
                    Volume:
                    {round(self.volume_mm,2):.2f} µl
                    Height:
                    {round(self.height_mm,2):.2f} mm
                    R²:
                    {round(self.r2,5)}
                    '''   
                )
        else:
            ret = 'No droplet!'
            if self.error:
                ret = f'{ret}\n{self.error}'
        
        return ret

    @staticmethod
    def delete_scale():
        settings = QSettings()
        settings.remove("droplet")

    # properties section, get returns the average, set feeds the rolling averager
    @property
    def angle_l(self):
        """ average of left tangent angle """
        if self.averaging:
            return self._angle_l_avg.average
        else: 
            return self._angle_l

    @angle_l.setter
    def angle_l(self, value):
        self._angle_l_avg._put(value)
        self._angle_l = value

    @property
    def angle_r(self):
        """ average of right tangent angle """
        if self.averaging:
            return self._angle_r_avg.average
        else:
            return self._angle_r

    @angle_r.setter
    def angle_r(self, value):
        self._angle_r_avg._put(value)
        self._angle_r = value

    @property
    def height(self):
        """ height of droplet in px """
        if self.averaging:
            return self._height_avg.average
        else:
            return self._height

    @height.setter
    def height(self, value):
        self._height_avg._put(value)
        self._height = value

    @property
    def volume(self):
        """ approx volume of droplet  in px^3 """
        if self.averaging:
            return self._volume_avg.average
        else:
            return self._volume

    @volume.setter
    def volume(self, value):
        self._volume_avg._put(value)
        self._volume = value

    # return values after converting to metric
    @property
    def height_mm(self):
        """ droplet height in mm 
        
        .. seealso:: :meth:`set_scale` 
        """
        if self.averaging:
            return self._height_avg.average * self.scale_px_to_mm
        else:
            return self._height * self.scale_px_to_mm

    @property
    def base_diam_mm(self):
        """ droplet contact surface width in mm

        .. seealso:: :meth:`set_scale` 
        """
        return self.base_diam * self.scale_px_to_mm

    @property
    def volume_mm(self):
        if self.averaging:
            return self._volume_avg.average * self.scale_px_to_mm**3
        else:
            return self._volume * self.scale_px_to_mm**3

    def set_scale(self, scale):
        """ set and store a scalefactor to calculate mm from pixels

        :param scale: the scalefactor to calculate mm from px
        
        .. seealso:: :meth:`camera_control.CameraControl.calib_size` 
        """
        logging.info(f"droplet: set scale to {scale}")
        self.scale_px_to_mm = scale
        # save in persistent storage
        settings = QSettings()
        settings.setValue("droplet/scale_px_to_mm", scale)

    def set_filter_length(self, value):
        """ adjust the filter length for the rolling average
        
        :param value: new filter length
        """
        self._angle_l_avg.set_length(value)
        self._angle_r_avg.set_length(value)
        self._height_avg.set_length(value)
        self._volume_avg.set_length(value)

    def reset_filters(self):
        """reset the filters for special modes
        """
        self._angle_l_avg.reset()
        self._angle_r_avg.reset()
        self._height_avg.reset()
        self._volume_avg.reset()

    def change_filter_mode(self, mode):
        """change the filter modes

        :param mode: modes: 0 default; 1 average until read
        """
        self._angle_l_avg.change_mode(mode)
        self._angle_r_avg.change_mode(mode)
        self._height_avg.change_mode(mode)
        self._volume_avg.change_mode(mode)

class RollingAverager:
    """ 
    rolling average filter of variable length
    
    :param length: length of the filter
    """
    def __init__(self, length=300):
        # lenght of filter
        self.length = length
        self.default_len = length
        self.buffer = [0.0]*length
        self.counter = 0
        self.mode = 0
        self.first_number = True

    def _rotate(self):
        """
        shift  internal index to next position
        """
        if self.mode == 0:
            # increase current index by 1 or loop back to 0
            if self.counter == (self.length - 1):
                self.counter = 0
            else:
                self.counter += 1

    def _put(self, value):
        """ set value at current index

        :param float value: the new value to set
        """ 
        if self.mode == 1:
            self.buffer.append(value)
            self.length += 1
        else:
            if self.first_number:
                # initialize buffer with first value
                self.buffer = [value]*self.length
                self.first_number = False
            else:
                # add value to current line then rotate index
                self.buffer[self.counter] = value
            self._rotate()

    @property
    def average(self) -> float:
        """ Return the averaged value """
        avg = sum(self.buffer) / self.length
        if self.mode == 1: 
            self.buffer = []
            self.length = 0
        return avg

    def set_length(self, value):
        """
        set the length of the filter, if new length is smaller the already exisitng values are cut off, if it is longer the current average is used to fille the ne spots

        :param int value: the new filter length
        """
        if value > self.length:
            # append delta len to exisiting buffer, fill w/ current average
            self.buffer = self.buffer + [self.average]*(value - self.length)
        else:
            # keep last numbers
            self.buffer = self.buffer[-value:]
            # roll counter over if too large
            if self.counter > (value -1 ): self.counter = 0
        self.length = value
        logging.info(f"set filter length to {value}")

    def reset(self):
        self.mode == 0
        self.counter = 0
        self.set_length(self.default_len)

    def change_mode(self, mode):
        self.mode = mode
        if mode == 0:
            self.reset()
        elif mode == 1:
            self.buffer = []
            self.length = 0
        else:
            self.set_length(1)
