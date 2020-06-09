from lip_reading.network.tf_based.layer import ResNet50
from tensorflow.keras import layers
import tensorflow as tf

def RES_BI_LSTM(nclasses=26,input_shape=(26,120,120,5)):

    input = layers.Input(shape=(26,120,120,5))

    splited_input = []
    for i in range(26):
        splited_input.append(layers.Lambda(lambda x: x[ :, i, :],input_shape=input_shape)(input))

    res = ResNet50()

    cnn_output = []
    for i in range(26):
        cnn_output.append(res(splited_input[i]))

    x = tf.stack(cnn_output,axis=1)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(512, return_sequences=True)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.LSTM(512)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(nclasses, activation='softmax')(x)

    model = tf.keras.Model(input,x)
    model.compile(optimizer="Adam", loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])
    model.summary()

    return model