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

from typing import List
import cv2
import numpy as np
import logging
import time

from PySide2 import QtGui
from PySide2.QtWidgets import QLabel, QOpenGLWidget
from PySide2.QtCore import  QTimer, Qt, QPoint, QRect, QSize, Slot
from PySide2.QtGui import QBrush, QImage, QPaintEvent, QPainter, QPen, QPixmap, QTransform
from needle_mask import DynamicNeedleMask

from resizable_rubberband import ResizableRubberBand
from baseline import Baseline
from evaluate_droplet import evaluate_droplet, ContourError
from droplet import Droplet

class CameraPreview(QOpenGLWidget):
    """ 
    widget to display camera feed and overlay droplet approximations from opencv
    """
    def __init__(self, parent=None):
        super(CameraPreview, self).__init__(parent)
        self.roi_origin = QPoint(0,0)
        self._pixmap: QPixmap = QPixmap(480, 360)
        self._double_buffer: QImage = None
        self._raw_image : np.ndarray = None
        self._image_size = (1,1)
        self._image_size_invalid = True
        self._roi_rubber_band = ResizableRubberBand(self)
        self._needle_mask = DynamicNeedleMask(self)
        self._needle_mask.update_mask_signal.connect(self.update_mask)
        self._baseline = Baseline(self)
        self._droplet = Droplet()
        self._mask = None

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(500)
        logging.debug("initialized camera preview")

    def prepare(self):
        """ preset the baseline to 250 which is roughly base of the test image droplet """
        # replaced by saving previous line height
        #self._baseline.y_level = self.mapFromImage(y=250)
        pass

    def paintEvent(self, event: QPaintEvent):
        """
        custom paint event to 
        draw camera stream and droplet approximation if available

        uses double buffering to avoid flicker
        """
        # completely override super.paintEvent() to use double buffering
        painter = QPainter(self)
        
        buf = self.doubleBufferPaint(self._double_buffer)
        # painting the buffer pixmap to screen
        painter.drawImage(0, 0, buf)
        painter.end()

    def doubleBufferPaint(self, buffer=None):
        self.blockSignals(True)
        #self.drawFrame(painter)
        if buffer is None:
            buffer = QImage(self.width(), self.height(), QImage.Format_RGB888)
        buffer.fill(Qt.black)
        # calculate offset and scale of droplet image pixmap
        scale_x, scale_y, offset_x, offset_y = self.get_from_image_transform()

        db_painter = QPainter(buffer)
        db_painter.setRenderHints(QPainter.Antialiasing | QPainter.NonCosmeticDefaultPen)
        db_painter.setBackground(QBrush(Qt.black))
        db_painter.setPen(QPen(Qt.black,0))
        db_painter.drawPixmap(offset_x, offset_y, self._pixmap)
        pen = QPen(Qt.magenta,1)
        pen_fine = QPen(Qt.blue,1)
        pen.setCosmetic(True)
        db_painter.setPen(pen)
        # draw droplet outline and tangent only if evaluate_droplet was successful
        if self._droplet.is_valid:
            try:
                # transforming true image coordinates to scaled pixmap coordinates
                db_painter.translate(offset_x, offset_y)
                db_painter.scale(scale_x, scale_y)

                # drawing tangents and baseline
                db_painter.drawLine(*self._droplet.line_l)
                db_painter.drawLine(*self._droplet.line_r)
                db_painter.drawLine(*self._droplet.int_l, *self._droplet.int_r)

                # move origin to ellipse origin
                db_painter.translate(*self._droplet.center)

                # draw diagnostics
                # db_painter.setPen(pen_fine)
                # #  lines parallel to coordinate axes
                # db_painter.drawLine(0,0,20*scale_x,0)
                # db_painter.drawLine(0,0,0,20*scale_y)
                # # angle arc
                # db_painter.drawArc(-5*scale_x, -5*scale_y, 10*scale_x, 10*scale_y, 0, -self._droplet.tilt_deg*16)

                # rotate coordinates to ellipse tilt
                db_painter.rotate(self._droplet.tilt_deg)

                # draw ellipse
                # db_painter.setPen(pen)
                db_painter.drawEllipse(-self._droplet.maj/2, -self._droplet.min/2, self._droplet.maj, self._droplet.min)
                
                # # major and minor axis for diagnostics
                # db_painter.drawLine(0, 0, self._droplet.maj/2, 0)
                # db_painter.drawLine(0, 0, 0, self._droplet.min/2)
            except Exception as ex:
                logging.error(ex)
        db_painter.end()
        self.blockSignals(False)
        return buffer

    def mousePressEvent(self,event):
        """
        mouse pressed handler
        
        creates ROI rubberband rectangle
        """
        if event.button() == Qt.LeftButton:
            # create new rubberband rectangle
            self.roi_origin = QPoint(event.pos())
            self._roi_rubber_band.hide()
            self._roi_rubber_band.setGeometry(QRect(self.roi_origin, QSize()))
            self._roi_rubber_band.show()

    def mouseMoveEvent(self, event):
        """
        mouse moved handler

        resizes the ROI rubberband rectangle if left mouse button is pressed
        """
        if event.buttons() == Qt.NoButton:
            pass
        elif event.buttons() == Qt.LeftButton:
            # resize rubberband while mouse is moving
            if not self.roi_origin.isNull():
                self._roi_rubber_band.setGeometry(QRect(self.roi_origin, event.pos()).normalized())
        elif event.buttons() == Qt.RightButton:
            pass

    def keyPressEvent(self, event):
        """
        keyboard pressed handler

        - Esc: aborts ROI rubberband and hides it
        - Enter: applys ROI from rubberband to camera
        """
        if event.key() == Qt.Key_Enter:
            # apply the ROI set by the rubberband
            self.parent().apply_roi()
        elif event.key() == Qt.Key_Escape:
            # hide rubberband
            self._abort_roi()
            self.update()

    @Slot(np.ndarray, bool)
    def update_image(self, cv_img: np.ndarray, eval: bool = True):
        """ 
        Updates the image_label with a new opencv image
        
        :param cv_img: camera image array
        :param eval: if True: do image processing on given image

        .. seealso:: :py:meth:`camera_control.CameraControl.update_image`
        """
        self._raw_image = cv_img
        try:
            # evaluate droplet only if camera is running or if a oneshot eval is requested
            if eval:
                try:
                    self._droplet.is_valid = False
                    #dt = time.time()
                    evaluate_droplet(cv_img, self.get_baseline_y(), self._mask)
                    #print(time.time() - dt)
                except (ContourError, cv2.error, TypeError) as ex:
                    self._droplet.error = str(ex)
                except Exception as ex:
                    logging.exception("Exception thrown in %s", "fcn:evaluate_droplet", exc_info=ex)
                    self._droplet.error = str(ex)
                finally:
                    #bloc
                    self._timer.start()
            # else:
            #     self._droplet.is_valid = False
            qt_img = self._convert_cv_qt(cv_img)
            self._pixmap = qt_img
            if self._image_size_invalid:
                self._image_size = np.shape(cv_img)
                self.set_new_baseline_constraints()
                self._image_size_invalid = False
            self.update()
            # del cv_img
        except Exception as ex:
            logging.exception("Exception thrown in %s", "class:camera_preview fcn:update_image", exc_info=ex)

    def grab_image(self, raw=False):
        if raw:
            return self._convert_cv_qt(self._raw_image, False)
        else:
            return self.doubleBufferPaint(self._double_buffer)

    def _convert_cv_qt(self, cv_img: np.ndarray, scaled=True):
        """
        Convert from an opencv image to QPixmap
        
        :param cv_img: opencv image as numpy array
        :param scaled: if true or omitted returns an image scaled to widget dimensions
        :returns: opencv image as full size QImage or QPixmap scaled to widget dimensions
        """
        #rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        #h, w, ch = rgb_image.shape
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(cv_img, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8)
        if scaled: 
            qimg_scaled = qimg.scaled(self.size(), aspectMode=Qt.KeepAspectRatio, mode=Qt.SmoothTransformation)
            return QPixmap.fromImage(qimg_scaled)
        else:
            return qimg

    def map_droplet_drawing_vals(self, droplet: Droplet):
        """ 
        convert the droplet values from image coords into pixmap coords and values better for drawing 
        
        :param droplet: droplet object containing the data
        :returns: **tuple** (tangent_l, tangent_r, int_l, int_r, center, maj, min)

            - **tangent_l**: start and end coordinates left tangent as (x1,y1,x2,y2)
            - **tangent_r**: start and end coordinates right tangent as (x1,y1,x2,y2)
            - **int_l**: left intersection of ellipse and baseline as (x,y)
            - **int_l**: right intersection of ellipse and baseline as (x,y)
            - **center**: center of the ellipse as (x,y)
            - **maj**: major axis length of the ellipse
            - **min**: minor axis length of the ellipse
        """
        tangent_l = tuple(self.mapFromImage(droplet.line_l[0:1]) + self.mapFromImage(droplet.line_l[2:3]))
        #tuple(map(lambda x: self.mapFromImage(*x), droplet.line_l))
        tangent_r = tuple(self.mapFromImage(droplet.line_r[0:1]) + self.mapFromImage(droplet.line_r[2:3])) 
        #tuple(map(lambda x: self.mapFromImage(*x), droplet.line_r))
        center = self.mapFromImage(*droplet.center)
        maj, min = self.mapFromImage(droplet.maj, droplet.min)
        int_l = self.mapFromImage(*droplet.int_l)
        int_r = self.mapFromImage(*droplet.int_r)
        return tangent_l, tangent_r, int_l, int_r, center, maj, min

    def mapToImage(self, x=None, y=None, w=None, h=None):
        """ 
        Convert QLabel coordinates to image pixel coordinates

        :param x: x coordinate to be transformed
        :param y: y coordinate to be transformed
        :returns: x or y or Tuple (x,y) of the transformed coordinates, depending on what parameters where given
        """
        scale_x, scale_y, offset_x, offset_y = self.get_from_image_transform()
        res: List[int] = []
        if x is not None:
            tr_x = int(round((x - offset_x) / scale_x))
            res.append(tr_x)
        if y is not None:
            tr_y = int(round((y - offset_y) / scale_y))
            res.append(tr_y)
        if w is not None:
            tr_w = int(round(w / scale_x))
            res.append(tr_w)
        if h is not None:
            tr_h = int(round(h / scale_y))
            res.append(tr_h)
        return tuple(res) if len(res)>1 else res[0]

    def mapFromImage(self, x=None, y=None, w=None, h=None):
        """ 
        Convert Image pixel coordinates to QLabel coordinates

        :param x: x coordinate to be transformed
        :param y: y coordinate to be transformed
        :returns: x or y or Tuple (x,y) of the transformed coordinates, depending on what parameters where given
        """
        scale_x, scale_y, offset_x, offset_y = self.get_from_image_transform()
        res: List[int] = []
        if x is not None:
            tr_x = int(round((x  * scale_x) + offset_x))
            res.append(tr_x)
        if y is not None:
            tr_y = int(round((y * scale_y) + offset_y))
            res.append(tr_y)
        if w is not None:
            tr_w = int(round(w  * scale_x))
            res.append(tr_w)
        if h is not None:
            tr_h = int(round(h  * scale_y))
            res.append(tr_h)
        return tuple(res) if len(res)>1 else res[0]

    def get_from_image_transform(self):
        """ 
        Gets the scale and offset for a Image to QLabel coordinate transform 

        :returns: 4-Tuple: Scale factors for x and y as tuple, Offset as tuple (x,y)
        """
        pw, ph = self._pixmap.size().toTuple()              # scaled image size
        ih, iw = self._image_size[0], self._image_size[1]   # original size of image
        cw, ch = self.size().toTuple()                      # display container size
        scale_x = float(pw / iw)
        offset_x = abs(pw - cw)/2
        scale_y = float(ph / ih)
        offset_y = abs(ph -  ch)/2
        return scale_x, scale_y, offset_x, offset_y

    def show_baseline(self):
        """ Show the baseline selector """
        self._baseline.show()

    def hide_baseline(self):
        """ Hide the baseline selector """
        self._baseline.hide()

    def hide_rubberband(self):
        """ Hide the rubberband """
        self._roi_rubber_band.hide()

    def get_baseline_y(self) -> int:
        """ 
        return the y value the baseline is on in image coordinates 
        
        :returns: y value of baseline in image coordinates
        """
        y_base = self._baseline.y_level
        y = self.mapToImage(y=y_base)
        return y

    def set_baseline_y(self, value):
        """ 
        set the y value the baseline is on in image coordinates 
        
        :returns: y value of baseline in image coordinates
        """
        y = self.mapFromImage(y=value)
        self._baseline.y_level = y
        return y

    def hide_mask(self):
        """hides and disables the needle mask
        """
        self._needle_mask.hide()
        self._mask = None

    def show_mask(self):
        """shows the needle mask
        """
        self._needle_mask.show()
        self.update_mask()

    def update_mask(self):
        """update mask from widget
        """
        mask_rect = self._needle_mask.get_mask_geometry()
        self._mask = self.mapToImage(*mask_rect[:])

    def get_mask_dim(self):
        rect = self.mapToImage(*self._needle_mask.get_mask_geometry())
        return rect

    def set_mask_dim(self, x,y,w,h):
        x,_,w,h = self.mapFromImage(x,y,w,h)
        self._needle_mask.set_mask_geometry(x,0,w,h)

    def set_new_baseline_constraints(self):
        """ set the min and max y value for the baseline """
        pix_size = self._pixmap.size()
        offset_y = int(round(abs(pix_size.height() - self.height())/2))
        self._baseline.max_level = pix_size.height() + offset_y
        self._baseline.min_level = offset_y

    def get_roi(self):
        """ return the ROI selected by the rubberband """
        if self._roi_rubber_band.isHidden():
            return None
        x,y = self._roi_rubber_band.mapToParent(QPoint(0,0)).toTuple()
        w,h = self._roi_rubber_band.size().toTuple()
        self.hide_rubberband()
        x,y,w,h = self.mapToImage(x,y,w,h)
        #w,h = self.mapToImage(x=w, y=h)
        return x,y,w,h

    def _abort_roi(self):
        """
        abort ROI set by hiding the rubberband selector
        """
        self._roi_rubber_band.hide()
        logging.info("aborted ROI select")

    def invalidate_imagesize(self):
        """
        invalidate image size, causes image size to be reevaluated on next camera image
        """
        self._image_size_invalid = True

    def invalidate_droplet(self):
        """invalidates droplet and thus hiding any overlay
        """
        self._droplet.is_valid = False