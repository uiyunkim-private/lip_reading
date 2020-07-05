import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import Model

from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, initializers, regularizers, metrics

def conv1_layer(x):
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)

    return x
def conv2_layer(x):
    x = MaxPooling2D((3, 3), 2)(x)

    shortcut = x

    for i in range(3):
        if (i == 0):
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x
def conv3_layer(x):
    shortcut = x

    for i in range(4):
        if (i == 0):
            x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x
def conv4_layer(x):
    shortcut = x

    for i in range(6):
        if (i == 0):
            x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x
def conv5_layer(x):
    shortcut = x

    for i in range(3):
        if (i == 0):
            x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x

def resnet():
    input_tensor = Input(shape=(600, 120, 1), dtype='float32', name='input')
    x = conv1_layer(input_tensor)
    x = conv2_layer(x)
    x = conv3_layer(x)
    x = conv4_layer(x)
    x = conv5_layer(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096)(x)
    model = Model(input_tensor, x)
    return model


def lip_reading_resnet():
    model = tf.keras.Sequential()
    model.add(Reshape(target_shape=(600,120,1),input_shape=(5,120,120,1)))
    model.add(resnet())
    model.add(Flatten())
    return model


def Lip_reading(num_classes=2):
    model = tf.keras.Sequential()
    model.add(TimeDistributed(lip_reading_resnet(),input_shape=(26,5,120,120,1)))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model
#
#
#
# def Res_Conv1():
#     model = tf.keras.Sequential()
#     #model.add(ZeroPadding2D(padding=(3, 3),input_shape=(120,120,3)))
#     model.add(ZeroPadding2D(padding=(3, 3)))
#     model.add(Conv2D(64, (7, 7), strides=(2, 2)))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(ZeroPadding2D(padding=(1, 1)))
#     return model
#
# def Res_Conv2():
#     model = tf.keras.Sequential()
#     model.add(MaxPooling2D((3, 3), 2))
#
#
#     shortcut = x
#
#     for i in range(3):
#         if (i == 0):
#             x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x)
#
#             x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x)
#
#             x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
#             shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)
#             x = BatchNormalization()(x)
#             shortcut = BatchNormalization()(shortcut)
#
#             x = Add()([x, shortcut])
#             x = Activation('relu')(x)
#
#             shortcut = x
#
#         else:
#             x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x)
#
#             x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
#             x = BatchNormalization()(x)
#             x = Activation('relu')(x)
#
#             x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
#             x = BatchNormalization()(x)
#
#             x = Add()([x, shortcut])
#             x = Activation('relu')(x)
#
#             shortcut = x
#
#     return x
#
# # class Res_Conv1(tf.keras.layers.Layer):
# #     def __init__(self):
# #         super(Res_Conv1, self).__init__()
# #
# #     def call(self, inputs, training=None, mask=None):
# #         x = inputs
# #
# #         #x = ZeroPadding2D(padding=(3, 3))(x)
# #
# #         x = Conv2D(64, (7, 7), strides=(2, 2))(x)
# #         x = BatchNormalization()(x)
# #         x = Activation('relu')(x)
# #         x = ZeroPadding2D(padding=(1, 1))(x)
# #
# #         return x
#
# class Res_Conv2(tf.keras.layers.Layer):
#     def __init__(self):
#         super(Res_Conv2, self).__init__()
#
#     def call(self, inputs, training=None, mask=None):
#         x = inputs
#         x = MaxPooling2D((3, 3), 2)(x)
#
#         shortcut = x
#
#         for i in range(3):
#             if (i == 0):
#                 x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
#                 x = BatchNormalization()(x)
#                 x = Activation('relu')(x)
#
#                 x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
#                 x = BatchNormalization()(x)
#                 x = Activation('relu')(x)
#
#                 x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
#                 shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)
#                 x = BatchNormalization()(x)
#                 shortcut = BatchNormalization()(shortcut)
#
#                 x = Add()([x, shortcut])
#                 x = Activation('relu')(x)
#
#                 shortcut = x
#
#             else:
#                 x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
#                 x = BatchNormalization()(x)
#                 x = Activation('relu')(x)
#
#                 x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
#                 x = BatchNormalization()(x)
#                 x = Activation('relu')(x)
#
#                 x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
#                 x = BatchNormalization()(x)
#
#                 x = Add()([x, shortcut])
#                 x = Activation('relu')(x)
#
#                 shortcut = x
#
#         return x
#
# class Res_Conv3(tf.keras.layers.Layer):
#     def __init__(self):
#         super(Res_Conv3, self).__init__()
#
#     def call(self, inputs, training=None, mask=None):
#         x = inputs
#         shortcut = inputs
#
#         for i in range(4):
#             if (i == 0):
#                 x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
#                 x = BatchNormalization()(x)
#                 x = Activation('relu')(x)
#
#                 x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
#                 x = BatchNormalization()(x)
#                 x = Activation('relu')(x)
#
#                 x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
#                 shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
#                 x = BatchNormalization()(x)
#                 shortcut = BatchNormalization()(shortcut)
#
#                 x = Add()([x, shortcut])
#                 x = Activation('relu')(x)
#
#                 shortcut = x
#
#             else:
#                 x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
#                 x = BatchNormalization()(x)
#                 x = Activation('relu')(x)
#
#                 x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
#                 x = BatchNormalization()(x)
#                 x = Activation('relu')(x)
#
#                 x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
#                 x = BatchNormalization()(x)
#
#                 x = Add()([x, shortcut])
#                 x = Activation('relu')(x)
#
#                 shortcut = x
#
#         return x
#
# class Res_Conv4(tf.keras.layers.Layer):
#     def __init__(self):
#         super(Res_Conv4, self).__init__()
#
#     def call(self, inputs, training=None, mask=None):
#         x = inputs
#         shortcut = inputs
#
#         for i in range(6):
#             if (i == 0):
#                 x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(x)
#                 x = BatchNormalization()(x)
#                 x = Activation('relu')(x)
#
#                 x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
#                 x = BatchNormalization()(x)
#                 x = Activation('relu')(x)
#
#                 x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
#                 shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut)
#                 x = BatchNormalization()(x)
#                 shortcut = BatchNormalization()(shortcut)
#
#                 x = Add()([x, shortcut])
#                 x = Activation('relu')(x)
#
#                 shortcut = x
#
#             else:
#                 x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
#                 x = BatchNormalization()(x)
#                 x = Activation('relu')(x)
#
#                 x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
#                 x = BatchNormalization()(x)
#                 x = Activation('relu')(x)
#
#                 x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
#                 x = BatchNormalization()(x)
#
#                 x = Add()([x, shortcut])
#                 x = Activation('relu')(x)
#
#                 shortcut = x
#
#         return x
#
# class Res_Conv5(tf.keras.layers.Layer):
#     def __init__(self):
#         super(Res_Conv5, self).__init__()
#
#     def call(self, inputs, training=None, mask=None):
#         x = inputs
#         shortcut = inputs
#
#         for i in range(3):
#             if (i == 0):
#                 x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)
#                 x = BatchNormalization()(x)
#                 x = Activation('relu')(x)
#
#                 x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
#                 x = BatchNormalization()(x)
#                 x = Activation('relu')(x)
#
#                 x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
#                 shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')(shortcut)
#                 x = BatchNormalization()(x)
#                 shortcut = BatchNormalization()(shortcut)
#
#                 x = Add()([x, shortcut])
#                 x = Activation('relu')(x)
#
#                 shortcut = x
#
#             else:
#                 x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
#                 x = BatchNormalization()(x)
#                 x = Activation('relu')(x)
#
#                 x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
#                 x = BatchNormalization()(x)
#                 x = Activation('relu')(x)
#
#                 x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
#                 x = BatchNormalization()(x)
#
#                 x = Add()([x, shortcut])
#                 x = Activation('relu')(x)
#
#                 shortcut = x
#
#         return x
#
# class Resnet50(tf.keras.layers.Layer):
#     def __init__(self):
#         super(Resnet50,self).__init__()
#
#         self.conv1 = Res_Conv1()
#         self.conv2 = Res_Conv2()
#         self.conv3 = Res_Conv3()
#         self.conv4 = Res_Conv4()
#         self.conv5 = Res_Conv5()
#
#         self.globalAveragePool2D6 = GlobalAveragePooling2D()
#
#         self.flat7 = Flatten()
#         self.fc8 = Dense(4096)
#
#     #def call(self, inputs, training=None, mask=None):
#
#     def compute_output_shape(self, input_shape):
#         #print(input_shape)
#         return (30,4096)
#
# def res():
#     model = tf.keras.Sequential()
#     #model.add(Input(shape=(120,120,3)))
#     model.add(Res_Conv1())
#     model.add(Res_Conv2())
#     model.add(Res_Conv3())
#     model.add(Res_Conv4())
#     model.add(Res_Conv5())
#
#     model.add(GlobalAveragePooling2D())
#
#     model.add(Flatten())
#     model.add(Dense(4096))
#     model.build(input_shape=(120, 120, 3))
#     model.summary()
#
#     return model
#
# def vgg():
#     model = tf.keras.Sequential()
#     model.add(Conv2D(input_shape=(120, 120, 3), filters=96,kernel_initializer='he_normal', kernel_size=(3, 3),strides=(1,1), activation="relu"))
#     model.add(BatchNormalization())
#     model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
#
#     model.add(Conv2D(filters=256, kernel_size=(3, 3),strides=(2,2),kernel_initializer='he_normal', activation="relu"))
#     model.add(BatchNormalization())
#     model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
#
#
#     model.add(Conv2D(filters=512, kernel_size=(3, 3),strides=(1,1),kernel_initializer='he_normal', activation="relu"))
#     model.add(BatchNormalization())
#     model.add(Conv2D(filters=512, kernel_size=(3, 3),strides=(1,1),kernel_initializer='he_normal', activation="relu"))
#     model.add(BatchNormalization())
#     model.add(Conv2D(filters=512, kernel_size=(3, 3),strides=(1,1),kernel_initializer='he_normal', activation="relu"))
#     model.add(BatchNormalization())
#     model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
#     model.add(Dense(4096))
#     model.add(Flatten())
#
#
#
#     return model
#
#
#
# # class Lip_reading(tf.keras.Model):
# #     def __init__(self,num_classes):
# #         super(Lip_reading, self).__init__()
# #
# #         self.resnet1 = TimeDistributed(Resnet50())
# #         #self.fc2 = TimeDistributed(Dense(4096))
# #
# #         self.bilstm3 = LSTM(512, return_sequences=True)
# #
# #         self.dropout3 = Dropout(0.5)
# #         self.bilstm4 = LSTM(512)
# #
# #         self.dropout4 = Dropout(0.5)
# #
# #         self.fc5 = Dense(num_classes, activation='softmax')
# #
# #     def call(self, inputs, training=None, mask=None):
# #
# #
# #         inputs = tf.split(inputs, num_or_size_splits=4, axis=0)
# #         #print(inputs)
# #         for input in inputs:
# #             input = tf.reshape(input,(30,120,120,3))
# #
# #
# #             frames = tf.split(input,num_or_size_splits=30,axis=0)
# #             outputs = []
# #             for frame in frames:
# #                 frame = tf.reshape(frame,(None,120,120,3))
# #                 outputs.append(self.resnet1(frame))
# #
# #             print(outputs)
# #
# #
# #
# #             x = self.bilstm3(x)
# #             x = self.dropout3(x)
# #
# #             x = self.bilstm4(x)
# #             x = self.dropout4(x)
# #
# #             x = self.fc5(x)
# #         return x