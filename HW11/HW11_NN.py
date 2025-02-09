import os
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, ReLU, MaxPooling2D, Flatten, Dense, Dropout, Softmax, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from keras import models
from sklearn.model_selection import train_test_split

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# create folder to save plots
path = './HW/HW11/Test_Images'
if not os.path.exists(path):
    os.makedirs(path)

# plot random sample
np.random.seed(23464631)

no_samples = x_train.shape[0]
idx = np.random.choice(no_samples)
image = x_train[idx]
label = y_train[idx]

plt.imshow(image, cmap='binary')
plt.title(f'label: {label}')
plt.savefig('./HW/HW11/Test_Images/fashionMNIST_sample.png')

# Normalize pixel values to [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

def build_CNN():      # --> 91,01 % Accuracy
    model = tf.keras.Sequential()   # define model

    model.add(Conv2D(filters=8, kernel_size=(3,3), input_shape=(28, 28, 1), padding='same', activation='relu')) # stage 1
    model.add(MaxPooling2D(pool_size=(2,2)))    # reduce dim from 28x28 to 14x14

    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')) # stage 2
    model.add(MaxPooling2D(pool_size=(2,2)))    # reduce dim from 14x14 to 7x7

    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')) # stage 3
    model.add(BatchNormalization())     # just added in attempt to get better accuracy

    model.add(Flatten())    # stage 4

    model.add(Dense(units=128, activation='relu'))  # stage 5
    model.add(Dropout(rate=0.2))

    model.add(Dense(units=10, activation='softmax'))    # stage 6

    return model

my_CNN = build_CNN()
my_CNN.summary()


# Split into 80% training and 20% validation set
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.2,random_state=42)

# configure model for training
step_size = 0.001     # default for Adam is 0.001
# step_size = 0.00001 --> Accuracy = 81.27%, classifies all images as "Dress", (with 10 epochs)
# step_size = 0.0001  --> Accuracy = 89.41%, classifies all images as "Coat", (with 20 epochs)
# step_size = 0.001   --> Accuracy = 90.88%, classifies all images as "Bag"
#   with BatchNorm, 10 epochs      = 90.19%, classifies most images as "Bag", Jacket --> "Trouser", gets "Shirt" right
#   with BatchNorm, 20 epochs      = 90.41%, classifies all images as "Bag", except for the shirt -> "Sandal"
#   with SGD w/ Momentum           = 90.52%, classifies all images as "Bag"
# step_size = 0.01    --> Accuracy = 86.30%

my_CNN.compile(optimizer=Adam(learning_rate=step_size), loss='sparse_categorical_crossentropy', metrics=['acc'])
#SGD(learning_rate=step_size, momentum=0.9)

# train model for 10 epochs
no_epochs = 10
CNN_history = my_CNN.fit(x=x_train, y=y_train, epochs=no_epochs, validation_data=(x_val,y_val))

# Evaluate the model
test_loss, test_acc = my_CNN.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Convert to TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(my_CNN)
tflite_model = converter.convert()

# Save the model
with open("HW/HW11/fashion_mnist.tflite", "wb") as f:
    f.write(tflite_model)

print("TensorFlow Lite model saved as 'fashion_mnist.tflite'")
