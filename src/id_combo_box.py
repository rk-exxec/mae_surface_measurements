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

from PySide2.QtCore import QSettings, Qt
from PySide2.QtWidgets import QComboBox, QMenu

from typing import Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from ui_form import Ui_main

class IdComboBox(QComboBox):
    def __init__(self, parent):
        super(IdComboBox, self).__init__(parent)
        self.settings = QSettings()
        self._first_show = True
        self.ui: Ui_main = None
        settings_obj = self.settings.value("id_combo_control/ids")
        self._ids: Dict[str, str] = settings_obj if settings_obj != None else {}
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        
    def closeEvent(self, event):
        # save ids
        self.settings.setValue("id_combo_control/ids", self._ids)
        return super().closeEvent(event)

    def delete_entries(self):
        self.settings.remove("id_combo_control")

    def showEvent(self, event):
        if self._first_show:
            self._first_show = False
            self.ui = self.window().ui
            self.clear()
            self.addItems(self._ids.keys())
            self.connect_signals()
        return super().showEvent(event)

    def connect_signals(self):
        self.lineEdit().returnPressed.connect(self.save_dropdown_entry)
        self.currentIndexChanged.connect(self.update_iron_precentage)
        self.customContextMenuRequested.connect(self.showMenu)
        self.ui.ironContentEdit.returnPressed.connect(self.ui.idCombo.update_id_iron)

    def showMenu(self, pos):
        menu = QMenu()
        clear_action = menu.addAction("Clear all", self.clear())
        action = menu.exec_(self.mapToGlobal(pos))

    def save_dropdown_entry(self):
        """dropdown edit text edited
        """
        self.add_dropdown_entry(self.currentText())

    def add_dropdown_entry(self, entry):
        if entry != "" and not entry in self._ids.keys():
            #self.addItem(entry)
            self._ids[entry] = self.ui.ironContentEdit.text()

    def update_iron_precentage(self):
        cur_text = self.currentText()
        if cur_text in self._ids.keys():
            self.ui.ironContentEdit.setText(self._ids[cur_text])

    def update_id_iron(self):
        self._ids[self.currentText()] = self.ui.ironContentEdit.text()