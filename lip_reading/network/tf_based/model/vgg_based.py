from lip_reading.network.tf_based.layer import Vggnet
from tensorflow.keras import layers
import tensorflow as tf

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def VGG_BI_LSTM(nclasses=26,input_shape=(26,120,120,5)):

    input = layers.Input(shape=(26,120,120,5))



    splited_input = []
    for i in range(26):
        splited_input.append(layers.Lambda(lambda x: x[ :, i, :],input_shape=input_shape)(input))

    vgg = Vggnet()

    cnn_output = []
    for i in range(26):
        cnn_output.append(vgg(splited_input[i]))

    x = tf.stack(cnn_output,axis=1)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(512, return_sequences=True)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.LSTM(512)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(nclasses, activation='softmax')(x)


    model = tf.keras.Model(input,x)

    model.compile(optimizer="Adam", loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])
    #
    # model.add(layers.LSTM(512,return_sequences=True))
    # model.add(layers.Activation('relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.LSTM(512))
    # model.add(layers.Activation('relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(nclasses,activation='softmax'))
    model.summary()

    return model


VGG_BI_LSTM()