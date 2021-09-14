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
from PySide2.QtWidgets import QWidget, QHBoxLayout
from PySide2.QtGui import QBrush, QPainter, QPainterPath, QPen
from PySide2.QtCore import QRectF, Qt, QPoint, QSettings
COLOR = Qt.green

class Baseline(QWidget):  
    """ 
    Widget: Horizontal line that can be dragged up and down
    """
    def __init__(self, parent=None):
        super(Baseline, self).__init__(parent)
        #self.setWindowFlags(Qt.SubWindow)
        self.settings = QSettings()
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setCursor(Qt.SizeVerCursor)
        self.origin = QPoint(0,0)
        self._first_show = True
        self._y_level = 0
        self._max_level: int = 10000
        self._min_level: int = 0
        self.show()
        logging.debug("initialized baseline widget")

    @property
    def y_level(self) -> int:
        """
        the current y coordinate relative to parent widget
        """
        # y1 = self.mapToParent(self.pos()).y()
        # y2 = self.height()
        return self._y_level

    @y_level.setter
    def y_level(self, level):
        """
        the current y coordinate relative to parent widget  

        makes sure the widget does not leave visible area and
        compensates for own height
        :param level: new y level relative to parent widget
        """
        self._y_level = level
        level -= self.height()/2
        if level > self._min_level and level < self._max_level:
            self.move(QPoint(self.x(), level))
        elif level < self._min_level:
            self.move(QPoint(self.x(), self._min_level))
        elif level > self._max_level:
            self.move(QPoint(self.x(), self._max_level))

    @property
    def max_level(self):
        """
        the max y coordinate of the visible area relative to parent
        """
        return self._max_level + self.height()/2

    @max_level.setter
    def max_level(self, level):
        """
        set max y coordinate of visible area relative to parent (eg to exclude black bars)
        """
        self._max_level = level - self.height()/2

    @property
    def min_level(self):
        """
        the min y coordinate of the visible area relative to parent
        """
        return self._min_level + self.height()/2

    @min_level.setter
    def min_level(self, level):
        """
        set min y coordinate of visible area relative to parent (eg to exclude black bars)
        """
        self._min_level = level - self.height()/2

    def paintEvent(self, event):
        """
        custom paint event to draw horizontal line and handles at current y level
        """
        super().paintEvent(event)
        painter = QPainter(self)
        #painter.beginNativePainting()
        #painter.setRenderHint(QPainter.Antialiasing)
        x1,y1,x2,y2 = self.rect().getCoords()
        #painter.setPen(QPen(Qt.gray, 1))
        #painter.drawRect(self.rect())
        painter.setPen(QPen(COLOR, 1))
        painter.drawLine(QPoint(x1, y1+(y2-y1)/2), QPoint(x2, y1+(y2-y1)/2))
        path = QPainterPath()
        path.moveTo(x1, y1)
        path.lineTo(x1, y1 + y2)
        path.lineTo(10, y1 + y2/2)
        path.lineTo(x1, y1)
        painter.fillPath(path, QBrush(COLOR))
        path = QPainterPath()
        path.moveTo(x2+1, y1)
        path.lineTo(x2+1, y1 + y2)
        path.lineTo(x2 - 9, y1 + y2/2)
        path.lineTo(x2+1, y1)
        painter.fillPath(path, QBrush(COLOR))

        #painter.endNativePainting()
        painter.end()

    def save_y_level(self):
        self.settings.setValue("baseline/y_level", self.y_level)

    def load_y_level(self):
        self.y_level = self.settings.value("baseline/y_level", defaultValue=0.0, type=float)

    def delete_y_level(self):
        self.settings.remove("baseline")

    def showEvent(self, event):
        """
        custom show event

        initializes geometry for first show
        """
        if self._first_show:
            self.setGeometry(0, self.parent().geometry().height() - 10, self.parent().geometry().width(), 20)
            self.max_level = self.parent().geometry().height() - self.height()
            self.load_y_level()
            self._first_show = False

    def resize(self, w: int, h: int) -> None:
        return super().resize(w, self.height())

    def rescalePos(self, scale_factor):
        self.y_level = self.y_level*scale_factor

    def mousePressEvent(self, event):
        """
        mouse button pressed handler

        remembers initial mouse down position
        """
        if event.button() == Qt.LeftButton:
            self.origin = event.globalPos() - self.pos()

    def mouseReleaseEvent(self, event):
        """
        mouse released handler

        remembers mouse up position
        """
        if event.button() == Qt.LeftButton:
            self.origin = event.pos()
            self.save_y_level()

    def mouseMoveEvent(self, event):
        """
        mouse moved handler

        adjusts the y level according to mouse movement
        """
        if event.buttons() == Qt.LeftButton:
            #new_y = event.globalPos().y() - self.origin.y()
            self.y_level = event.globalPos().y() - self.origin.y() + self.height()/2