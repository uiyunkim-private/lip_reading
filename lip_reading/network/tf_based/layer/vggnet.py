import tensorflow as tf
from tensorflow.keras import layers

def Vggnet():

    model = tf.keras.Sequential()

    model.add(layers.Conv2D(96, (3, 3), strides=1,input_shape=(120,120,5)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(256, (3, 3), strides=2))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(512, (3, 3), strides=1))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), strides=1))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), strides=1))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Dense(4096))
    model.add(layers.Flatten())

    return model
