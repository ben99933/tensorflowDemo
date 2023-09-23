import random as random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.utils import to_categorical # one-hot encoding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist

# =============================資料處理=============================================
(x_train, y_train), (x_test, y_test) = mnist.load_data()
n = 9487
# plt.imshow(x_train[n], cmap='Greys')
# plt.show()
x_train = x_train.reshape(60000, 28*28)/255
x_test = x_test.reshape(10000, 28*28)/255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# =============================建立模型=============================================
model = Sequential()
# add layer
# Dense: 全連接層
# input_dim: 輸入層
# activation: 激活函數
# activation: relu, sigmoid, softmax, tanh
# relu = max(0, x) 線性整流函數
model.add(Dense(100, input_dim=784, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax')) # output
model.compile(loss='mse', optimizer=SGD(lr=0.09), metrics=['accuracy'])
model.summary()
# =============================訓練模型=============================================
# batch_size: 每次訓練的筆數
# epochs: 訓練次數
model.fit(x_train, y_train, batch_size=100, epochs=20)
model.save("my_nn.h5")
# =============================評估模型=============================================
predict = model.predict_classes(x_test)

# 預測10次
for i in range(10):
    n = random.randint(0, 9999)
    score = model.evaluate(x_test, y_test)
    print('test loss:', score[0])
    print('test accuracy:', score[1])
    print('神經網路預測是:', predict[n])
    plt.imshow(x_test[n].reshape(28, 28), cmap='Greys')
    plt.show()