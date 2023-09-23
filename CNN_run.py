# 如果 my_cnn.h存在則載入模型
import os

from matplotlib import pyplot as plt
from tensorflow.python.keras.datasets import mnist
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

model = None
if os.path.exists('my_cnn.h5'):
    model = load_model('my_cnn.h5')
else:
    print('model not found')
    exit()

# 評估模型
scores = model.evaluate(x_test_normalize, y_test_onehot)
print('accuracy=', scores[1])

# 預測
prediction = model.predict_classes(x_test_normalize)
for i in range(10):
    print('prediction:', prediction[i])
    print('label:', y_test[i])
    plt.imshow(x_test[i], cmap='Greys')
    plt.show()
