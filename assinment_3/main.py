#使用全连接层实现手写数字识别的模型构造  编译  训练  预测 等步骤
'''
全连接代码如下：
:构建模型,中间层为121个神经元的全连接层,输入数据维度为28*28,使用激活函数为relu,最后分成10类进行输出,输出的激活函数使用softmax(分类器输出专用激活函数)
model = sequential()
model.add(Dense(units=121,input_dim=28*28))
model.add(Activation('relu'))
model.add(Dense(units = 10))
model.add(Activation('softmax'))
'''
import numpy as np
import keras
from keras.models import Sequential,load_model
from keras.layers import Conv2D, MaxPooling2D,Flatten,Dense,Dropout
import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 

os.environ["CUDA_VISIBLE_DEVICES"]="0" 
print("Current device:", tf.test.gpu_device_name())
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
    return x_train.reshape(-1,28,28,1)


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
model.add(
    Conv2D(128,kernel_size=(3,3),activation='relu',input_shape=(28,28,1))
)
model.add(Conv2D(64,(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))
model.add(Flatten())#链接层，卷积与全连接的中介
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))

#编译模型，指定loss函数和优化器
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#训练模型
#batch_size  一次抓取的样本数量   60000 / 64 = 938
#epoch将所有样本训练几轮
with tf.device("/gpu:0"):
    model.fit(x_train,
            y_train,
            batch_size=128,
            epochs=10,
            verbose=1,
            validation_data=(x_test, y_test))
#验证模型
#verbose参数用于带进度条的输出信息，1为输出，0为不输出
score = model.evaluate(x_test, y_test, verbose=0)
#保存模型
model.save('keras_minist.model')
#评估
print('loss:', score[0])
print('accuracy', score[1])
