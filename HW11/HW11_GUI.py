import sys

from PyQt5.QtWidgets import (QApplication, QPushButton, QWidget,QMainWindow,
                             QFileDialog, QGridLayout, QLabel, QVBoxLayout, QHBoxLayout, QMessageBox, QComboBox)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

import pyqtgraph as pg
import json
import imageio.v2 as io

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
    
        # Run function to set defaults
        # self.set_defaults()

        self.setWindowTitle("Image Classifier")
        self.setGeometry(100, 100, 600, 400)

        # create central widget
        c_widget = QWidget(self)
        self.setCentralWidget(c_widget)
        self.mainLayout = QVBoxLayout()
        c_widget.setLayout(self.mainLayout)

        # Image display label
        self.image_label = QLabel("Drag an image here", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #aaa;")
        self.image_label.setFixedSize(400, 300)
        
        self.mainLayout.addWidget(self.image_label)

        # Classification result label
        self.result_label = QLabel("Classification result: ", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.mainLayout.addWidget(self.result_label)
        
        # Buttons
        self.classify_button = QPushButton("Classify", self)
        self.classify_button.clicked.connect(self.classify_image)
        self.mainLayout.addWidget(self.classify_button)
        
        self.clear_button = QPushButton("Clear Image", self)
        self.clear_button.clicked.connect(self.clear_image)
        self.mainLayout.addWidget(self.clear_button)

        # Enable drag-and-drop functionality
        self.setAcceptDrops(True)
        self.file_path = None


    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.display_image(file_path)
    
    def display_image(self, file_path):
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        else:
            self.image_label.setText("Failed to load image")

    def classify_image(self):
        if self.file_path:
            # Placeholder for actual classification logic
            self.result_label.setText(f"Classification result: Processing {self.file_path}")
        else:
            self.result_label.setText("No image loaded")
    
    def clear_image(self):
        self.image_label.setText("Drag an image here")
        self.image_label.setPixmap(QPixmap())
        self.result_label.setText("Classification result: ")
        self.file_path = None



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
