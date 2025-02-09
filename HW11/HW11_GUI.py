import sys
import numpy as np
import tensorflow.lite as tflite

from PyQt5.QtWidgets import (QApplication, QPushButton, QWidget,QMainWindow,
                             QFileDialog, QLabel, QVBoxLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2

# Load class names for Fashion MNIST
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Classifier for Clothes")
        self.setGeometry(100, 100, 300, 500)

        # create central widget
        c_widget = QWidget(self)
        self.setCentralWidget(c_widget)
        self.mainLayout = QVBoxLayout()
        c_widget.setLayout(self.mainLayout)

        self.original_image_label = QLabel("Drag an image here or click to select", self)
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setStyleSheet("border: 2px dashed #aaa;")
        self.original_image_label.setFixedSize(300, 300)     # (400, 300)
        self.mainLayout.addWidget(self.original_image_label)
        self.original_image_label.mousePressEvent = self.open_file_dialog

        # classification result
        self.result_label = QLabel("Classification result: ", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.mainLayout.addWidget(self.result_label)

        # BUTTONS
        # Classify-Button
        self.classify_button = QPushButton("Classify", self)
        self.classify_button.clicked.connect(self.classify_image)
        self.mainLayout.addWidget(self.classify_button)

        # Clear Image-Button
        self.clear_button = QPushButton("Clear Image", self)
        self.clear_button.clicked.connect(self.clear_image)
        self.mainLayout.addWidget(self.clear_button)

        # LOAD TFLITE
        # Load TensorFlow Lite model
        self.interpreter = tflite.Interpreter(model_path="./HW/HW11/fashion_mnist.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        #DRAG-AND-DROP
        # Enable drag-and-drop functionality
        self.setAcceptDrops(True)
        self.file_path = None
        self.image_loaded = False


    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if not self.image_loaded:
            urls = event.mimeData().urls()
            if urls:
                self.file_path = urls[0].toLocalFile()
                self.display_image(self.file_path)

    def open_file_dialog(self, event):
        if not self.image_loaded:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
            if file_path:
                self.file_path = file_path
                self.display_image(self.file_path)

    def display_image(self, file_path):
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            self.original_image_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.image_loaded = True
        else:
            self.original_image_label.setText("Failed to load image")

    def preprocess_image(self, file_path):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))     # ensure aspect ratio is fine while resizing
        img = img.astype(np.float32) / 255.0  # Normalize pixel values
        expected_shape = self.input_details[0]['shape']  # Get expected model input shape
        img = img.reshape(expected_shape)  # reshape dynamically, ensure shape is correct
        return img

    def numpy_to_qpixmap(self, img):
        img = (img.squeeze() * 255).astype(np.uint8)  # convert back to 0-255 pixel range
        qimg = QImage(img, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg)

    def classify_image(self):
        if self.file_path:
            img = self.preprocess_image(self.file_path)

            if img is None:
                self.result_label.setText("Error loading image")
                return
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], img)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

       # predicted_label = np.argmax(probabilities)
            predicted_label = np.argmax(output_data)
            self.result_label.setText(f"Classification result: {class_names[predicted_label]}")
            
        else:
            self.result_label.setText("No image loaded")
    
    def clear_image(self):
        self.original_image_label.setText("Original Image")
        self.original_image_label.setPixmap(QPixmap())
        self.result_label.setText("Classification result: ")
        self.file_path = None
        self.image_loaded = False



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
