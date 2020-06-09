#from lip_reading.network.tf_based.layer import ResNet50
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
def BI_LSTM(nclasses=26,input_shape=(26,120,120,5)):

    model = tf.keras.Sequential()
    model.add(layers.TimeDistributed(ResNet50(include_top=False,weights=None),input_shape=input_shape))
    model.add(layers.TimeDistributed(layers.Dense(4086)))
    model.add(layers.TimeDistributed(layers.Flatten()))

    model.add(layers.LSTM(512,return_sequences=True))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.LSTM(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(nclasses,activation='softmax'))
    model.summary()

    return model