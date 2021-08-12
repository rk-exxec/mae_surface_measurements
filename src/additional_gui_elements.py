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
import math

from camera import AbstractCamera

from PySide2 import QtWidgets
from PySide2.QtCore import QCoreApplication, QObject, QRect, QSize, Slot, QTimer
from PySide2.QtGui import QFont, Qt
from PySide2.QtWidgets import QDialog, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSizePolicy, QSlider, QSpinBox, QVBoxLayout, QWidget


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super(AboutDialog, self).__init__(parent)
        self.setupUI()

        # Add button signal to greetings slot
        self.okButton.clicked.connect(self.hide)

        self.show()


    def setupUI(self):
        self.resize(431, 260)
        self.licenseText = QPlainTextEdit(self)
        self.licenseText.setObjectName(u"licenseText")
        self.licenseText.setEnabled(True)
        self.licenseText.setGeometry(QRect(10, 10, 411, 211))
        self.licenseText.setFrameShape(QFrame.StyledPanel)
        self.licenseText.setFrameShadow(QFrame.Sunken)
        self.licenseText.setUndoRedoEnabled(False)
        self.licenseText.setTextInteractionFlags(Qt.NoTextInteraction)
        self.okButton = QPushButton(self)
        self.okButton.setObjectName(u"okButton")
        self.okButton.setGeometry(QRect(350, 230, 75, 23))

        self.retranslateUi()
    # setupUi

    def retranslateUi(self):
        self.setWindowTitle(QCoreApplication.translate("about", u"About MAEsure", None))
        self.licenseText.setPlainText(QCoreApplication.translate("about", u"MAEsure is a program to measure the surface energy of MAEs via contact angle\n"
"Copyright (C) 2021  Raphael Kriegl\n"
"\n"
"This program is free software: you can redistribute it and/or modify\n"
"it under the terms of the GNU General Public License as published by\n"
"the Free Software Foundation, either version 3 of the License, or\n"
"(at your option) any later version.\n"
"\n"
"This program is distributed in the hope that it will be useful,\n"
"but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
"MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
"GNU General Public License for more details.\n"
"\n"
"You should have received a copy of the GNU General Public License\n"
"along with this program.  If not, see <https://www.gnu.org/licenses/>.", None))
        self.licenseText.setPlaceholderText("")
        self.okButton.setText(QCoreApplication.translate("about", u"OK", None))
    # retranslateUi

class CamInfoDialog(QDialog):
    def __init__(self, parent, cam):
        super(CamInfoDialog, self).__init__(parent=parent)
        self.cam = cam
        self.setupUI()
        
        # Add button signal to greetings slot
        self.okButton.clicked.connect(self.hide)

        self.show()


    def setupUI(self):
        self.resize(431, 260)
        self.licenseText = QPlainTextEdit(self)
        self.licenseText.setObjectName(u"camInfo")
        self.licenseText.setEnabled(True)
        self.licenseText.setGeometry(QRect(10, 10, 411, 211))
        self.licenseText.setFrameShape(QFrame.StyledPanel)
        self.licenseText.setFrameShadow(QFrame.Sunken)
        self.licenseText.setUndoRedoEnabled(False)
        self.licenseText.setTextInteractionFlags(Qt.NoTextInteraction)
        self.licenseText.setFont(QFont("Courier New"))
        self.okButton = QPushButton(self)
        self.okButton.setObjectName(u"okButton")
        self.okButton.setGeometry(QRect(350, 230, 75, 23))

        self.retranslateUi()
    # setupUi

    def retranslateUi(self):
        self.setWindowTitle(QCoreApplication.translate("about", u"Camera Info", None))
        self.licenseText.setPlainText(QCoreApplication.translate("about", str(self.cam)))
        self.licenseText.setPlaceholderText("")
        self.okButton.setText(QCoreApplication.translate("about", u"OK", None))
    # retranslateUi

    
class CameraSettings(QDialog):
    def __init__(self, parent, cam) -> None:
        super().__init__(parent=parent)
        self.setupUI()

        self.timer = QTimer(self)
        self.timer.setSingleShot(False)
        self.timer.setInterval(500)
        self.auto_update = False
        self.connect_signals()
        
        self.cam: AbstractCamera = cam

        self.exp_value.setValue(self.cam.get_exposure())
        mode = self.cam.get_exposure_mode().as_tuple()
        if mode[0] == "Off":
            self.exp_off_btn.setEnabled(False)
            self.exp_cont_btn.setEnabled(True)
            self.exp_slider.setEnabled(True)
            self.exp_value.setEnabled(True)
            self.auto_update = False
        elif mode[0] == "Once":
            self.set_exp_mode("Off")
        elif mode[0] == "Continuous":
            self.exp_cont_btn.setEnabled(False)
            self.exp_off_btn.setEnabled(True)
            self.exp_slider.setEnabled(False)
            self.exp_value.setEnabled(False)
            self.auto_update = True

        range = self.cam.get_exposure_range()
        self.exp_value.setRange(*range)
        self.exp_slider.setRange(math.ceil(math.log10(range[0])*100), math.floor(math.log10(range[1])*100))
        self.timer.start()

    def connect_signals(self):
        self.exp_slider.sliderMoved.connect(self.slider_val_changed)
        self.exp_value.valueChanged.connect(self.value_changed)
        #self.exp_slider.sliderReleased.connect(self.update_exposure)
        self.exp_off_btn.clicked.connect(lambda: self.set_exp_mode("Off"))
        self.exp_once_btn.clicked.connect(lambda: self.set_exp_mode("Once"))
        self.exp_cont_btn.clicked.connect(lambda: self.set_exp_mode("Continuous"))
        self.rough_exp_btn.clicked.connect(lambda: self.set_exposure(8000))
        self.one_thousand_btn.clicked.connect(lambda: self.set_exposure(1000))
        self.timer.timeout.connect(self.update_exposure)

    @Slot(int)
    def slider_val_changed(self, value):
        self.exp_value.setValue(10**(value/100))

    @Slot(str)
    def value_changed(self, value):
        self.exp_slider.setValue(math.log10(value)*100)
        self.set_exposure(value)

    def set_exposure(self, value):
        if(not value): value = self.exp_value.value()
        logging.debug(f"Setting camera exposure to {value}")
        self.cam.set_exposure(value)

    @Slot()
    def update_exposure(self):
        if self.auto_update:
            self.exp_value.setValue(self.cam.get_exposure())
        
    @Slot(str)
    def set_exp_mode(self, mode):
        logging.debug(f"Setting camera exposure mode to {mode}")
        self.cam.set_exposure_mode(mode)
        if mode == "Off":
            self.exp_off_btn.setEnabled(False)
            self.exp_cont_btn.setEnabled(True)
            self.exp_slider.setEnabled(True)
            self.exp_value.setEnabled(True)
            self.auto_update = False
        elif mode == "Once":
            self.set_exp_mode("Off")
        elif mode == "Continuous":
            self.exp_cont_btn.setEnabled(False)
            self.exp_off_btn.setEnabled(True)
            self.exp_slider.setEnabled(False)
            self.exp_value.setEnabled(False)
            self.auto_update = True
            
        

    def setupUI(self):
        self.setObjectName(u"CustomCamera")
        self.resize(359, 120)
        self.verticalLayoutWidget = QWidget(self)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(10, 0, 341, 111))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(self.verticalLayoutWidget)
        self.label.setObjectName(u"label")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setAlignment(Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.label)

        self.exp_slider = QSlider(self.verticalLayoutWidget)
        self.exp_slider.setObjectName(u"exp_slider")
        sizePolicy1 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.exp_slider.sizePolicy().hasHeightForWidth())
        self.exp_slider.setSizePolicy(sizePolicy1)
        self.exp_slider.setMinimumSize(QSize(150, 0))
        self.exp_slider.setMaximum(700)
        self.exp_slider.setSingleStep(1)
        self.exp_slider.setPageStep(1)
        self.exp_slider.setValue(30)
        self.exp_slider.setOrientation(Qt.Horizontal)
        self.exp_slider.setInvertedControls(False)
        self.exp_slider.setTickPosition(QSlider.TicksAbove)
        self.exp_slider.setTickInterval(10)

        self.horizontalLayout.addWidget(self.exp_slider)

        self.exp_value = QDoubleSpinBox(self.verticalLayoutWidget)
        self.exp_value.setObjectName(u"exp_value")
        self.exp_value.setRange(1, 1e7)
        self.exp_value.setDecimals(1)
        self.exp_value.setButtonSymbols(QDoubleSpinBox.NoButtons)
        sizePolicy2 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.exp_value.sizePolicy().hasHeightForWidth())
        self.exp_value.setSizePolicy(sizePolicy2)

        self.horizontalLayout.addWidget(self.exp_value)

        self.label_2 = QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout.addWidget(self.label_2)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.exp_off_btn = QPushButton(self.verticalLayoutWidget)
        self.exp_off_btn.setObjectName(u"exp_off_btn")
        self.exp_off_btn.setCheckable(False)
        self.exp_off_btn.setChecked(False)
        self.exp_off_btn.setAutoExclusive(False)

        self.horizontalLayout_4.addWidget(self.exp_off_btn)

        self.exp_once_btn = QPushButton(self.verticalLayoutWidget)
        self.exp_once_btn.setObjectName(u"exp_once_btn")
        self.exp_once_btn.setCheckable(False)
        self.exp_once_btn.setAutoExclusive(False)

        self.horizontalLayout_4.addWidget(self.exp_once_btn)

        self.exp_cont_btn = QPushButton(self.verticalLayoutWidget)
        self.exp_cont_btn.setObjectName(u"exp_cont_btn")
        sizePolicy3 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.exp_cont_btn.sizePolicy().hasHeightForWidth())
        self.exp_cont_btn.setSizePolicy(sizePolicy3)
        self.exp_cont_btn.setMinimumSize(QSize(0, 0))
        self.exp_cont_btn.setCheckable(False)
        self.exp_cont_btn.setAutoExclusive(False)

        self.horizontalLayout_4.addWidget(self.exp_cont_btn)
        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")

        self.one_thousand_btn = QPushButton(self.verticalLayoutWidget)
        self.one_thousand_btn.setObjectName(u"one_thousand_btn")
        sizePolicy4 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.one_thousand_btn.sizePolicy().hasHeightForWidth())
        self.one_thousand_btn.setSizePolicy(sizePolicy4)
        self.one_thousand_btn.setMinimumSize(QSize(0, 0))
        self.one_thousand_btn.setCheckable(False)
        self.one_thousand_btn.setAutoExclusive(False)
        self.horizontalLayout_5.addWidget(self.one_thousand_btn)

        self.rough_exp_btn = QPushButton(self.verticalLayoutWidget)
        self.rough_exp_btn.setObjectName(u"rough_exp_btn")
        sizePolicy5 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.rough_exp_btn.sizePolicy().hasHeightForWidth())
        self.rough_exp_btn.setSizePolicy(sizePolicy5)
        self.rough_exp_btn.setMinimumSize(QSize(0, 0))
        self.rough_exp_btn.setCheckable(False)
        self.rough_exp_btn.setAutoExclusive(False)
        self.horizontalLayout_5.addWidget(self.rough_exp_btn)


        self.verticalLayout.addLayout(self.horizontalLayout_5)

        self.retranslateUi()

    # setupUi

    def retranslateUi(self):
        self.setWindowTitle(QCoreApplication.translate("CameraSettings", u"CameraSettings", None))
        self.label.setText(QCoreApplication.translate("CameraSettings", u"Exposure:", None))
        self.label_2.setText(QCoreApplication.translate("CameraSettings", u"\u00b5S", None))
        self.exp_off_btn.setText(QCoreApplication.translate("CameraSettings", u"Off", None))
        self.exp_once_btn.setText(QCoreApplication.translate("CameraSettings", u"Once", None))
        self.exp_cont_btn.setText(QCoreApplication.translate("CameraSettings", u"Continuous", None))
        self.one_thousand_btn.setText(QCoreApplication.translate("CameraSettings", u"1000", None))
        self.rough_exp_btn.setText(QCoreApplication.translate("CameraSettings", u"8000", None))
