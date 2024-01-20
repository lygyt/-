#使用全连接层实现手写数字识别的模型构造  编译  训练  预测 等步骤
'''
核心代码如下：
:构建模型,中间层为121个神经元的全连接层,输入数据维度为28*28,使用激活函数为relu,最后分成10类进行输出,输出的激活函数使用softmax(分类器输出专用激活函数)
model = sequential()
model.add(Dense(units=121,input_dim=28*28))
model.add(Activation('relu'))
model.add(Dense(units = 10))
model.add(Activation('softmax'))
'''
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation


#####################################################################################
#数据处理部分
#定义函数read_label  read_image
def read_label(add, nums):
    y_train = np.zeros(nums)
    with open(add, 'rb') as file_label:
        file_label.seek(8)
        data = file_label.read(nums)

        for i in range(nums):
            y_train[i] = data[i]
        # y_train 处理结束
        return y_train


def read_image(add, nums):
    x_train = np.zeros(nums * 28 * 28)
    with open(add, 'rb') as image:
        image.seek(16)
        data = image.read(nums * 28 * 28)
        for i in range(nums * 28 * 28):
            x_train[i] = data[i] / 255
    # x_train 处理结束
    return x_train.reshape(-1, 28 * 28)


#读取60000张训练数据
x_train = read_image(
    r'assinment_3\raw\train-images.idx3-ubyte', 60000)
y_train = read_label(
    r'assinment_3\raw\train-labels.idx1-ubyte', 60000)
#读取10000张测试数据
x_test = read_image(r'assinment_3\raw\t10k-images.idx3-ubyte',
                    10000)
y_test = read_label(r'assinment_3\raw\t10k-labels.idx1-ubyte',
                    10000)
#数据处理部分结束
#####################################################################################
#模型训练部分

#one-hot转码,分成几类后面就写几
#one-hot编码用于解决标签数据产生的误解问题
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#进行构造模型，使用keras
model = Sequential()
model.add(Dense(units=121, input_dim=28 * 28))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

#编译模型，指定loss函数和优化器
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

#训练模型
#batch_size  一次抓取的样本数量   60000 / 64 = 938
#epoch将所有样本训练几轮
model.fit(x_train,
          y_train,
          batch_size=64,
          epochs=30,
          verbose=1,
          validation_data=(x_test, y_test))
#验证模型
#verbose参数用于带进度条的输出信息，1为输出，0为不输出
score = model.evaluate(x_test, y_test, verbose=1)
model.save('assignment_3/keras_minist_dense.model')
print('loss:', score[0])
print('accuracy', score[1])
