import sys

from PyQt5.QtWidgets import (QApplication, QPushButton, QWidget,QMainWindow,
                             QFileDialog, QGridLayout, QLabel, QVBoxLayout, QHBoxLayout, QMessageBox, QComboBox)

import pyqtgraph as pg
import json
import imageio.v2 as io

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
    
        # Run function to set defaults
        # self.set_defaults()

        # create central widget
        c_widget = QWidget(self)
        self.setCentralWidget(c_widget)
        self.mainLayout = QVBoxLayout()
        c_widget.setLayout(self.mainLayout)



# to do:
# - write functions:
#       - set_defaults()