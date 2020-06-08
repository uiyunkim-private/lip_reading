from lip_reading.network.tf_based.layer import ResNet50
from tensorflow.keras import layers
import tensorflow as tf

def BI_LSTM(nclasses=26,input_shape=(26,120,120,5)):

    model = tf.keras.Sequential()
    model.add(layers.TimeDistributed(ResNet50(),input_shape=input_shape))
    model.add(layers.Dropout(0.5))
    model.add(layers.Bidirectional(layers.LSTM(512,return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(512)))
    model.add(layers.Dense(nclasses,activation='softmax'))
    model.summary()

    return model