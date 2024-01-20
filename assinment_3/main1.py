import numpy as np
import keras
from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


def read_labels(filename, items):  # 读取图片标记，也就是要学习的数字
    file_labels = open(filename, 'rb')
    file_labels.seek(8)  # 标签文件的头是8个字节，略过不读
    data = file_labels.read(items)
    y = np.zeros(items)
    for i in range(items):
        y[i] = data[i]
    file_labels.close()
    return y


y_train = read_labels(r'assinment_3\raw\train-labels.idx1-ubyte', 60000)  # 读取60000张训练标记
y_test = read_labels(r'assinment_3\raw\t10k-labels.idx1-ubyte', 10000)  # 读取10000张测试标记


def read_images(filename, items):  # 读取图片
    file_image = open(filename, 'rb')
    file_image.seek(16)

    data = file_image.read(items * 28 * 28)

    X = np.zeros(items * 28 * 28)
    for i in range(items * 28 * 28):
        X[i] = data[i] / 255
    file_image.close()
    return X.reshape(-1, 28, 28, 1)  # 请注意最后这一行，形状要整形成适合卷积网络的输入


X_train = read_images(r'assinment_3\raw\train-images.idx3-ubyte', 60000)  # 读取60000张训练图片
X_test = read_images(r'assinment_3\raw\t10k-images.idx3-ubyte', 10000)  # 读取10000张测试图片

y_train = keras.utils.to_categorical(y_train, 10)  # one hot转码
y_test = keras.utils.to_categorical(y_test, 10)  # one hot转码

# 训练与验证部分

model = Sequential()
model.add(
    Conv2D(32, kernel_size=(3, 3), activation='relu',
           input_shape=(28, 28, 1)))  # 32核卷积
model.add(MaxPooling2D(pool_size=(2, 2)))  # 2*2池化
model.add(Conv2D(64, (3, 3), activation='relu'))  # 64核卷积
model.add(MaxPooling2D(pool_size=(2, 2)))  # 2*2池化
model.add(Flatten())  # 连接层，卷积和全连接网络的中介
model.add(Dense(128, activation='relu'))  # 全连接层
model.add(Dropout(0.5))  # Dropout层
model.add(Dense(10, activation='softmax'))

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy'])  # 编译

model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=10,
    verbose=1,
    validation_data=(X_test, y_test))  # 训练
score = model.evaluate(X_test, y_test, verbose=0)  # 验证
print('损失数:', score[0])
print('准确率:', score[1])
