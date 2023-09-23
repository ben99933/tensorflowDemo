
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_4d = x_train.reshape(-1, 28, 28, 1).astype('float32')
x_test_4d = x_test.reshape(-1, 28, 28, 1).astype('float32')

# 正規化
x_train_normalize = x_train_4d / 255
x_test_normalize = x_test_4d / 255


# one-hot encoding
y_train_onehot = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)

n = 10000

model = Sequential()
# 卷積層 池化層
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 卷積層2 池化層2
model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # 扁平化: 36 * 7 * 7 => 1764*1
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax')) # output

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# 訓練
x_train_history = model.fit(x=x_train_normalize, y=y_train_onehot, validation_split=0.2, epochs=10, batch_size=300, verbose=2 )
model.save("my_cnn.h5")