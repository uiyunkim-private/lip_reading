from tensorflow.keras import layers
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
from tensorflow.keras import Model
import tensorflow as tf
from natsort import natsorted
import cv2
import os
import numpy as np

def lip_reading_image_processing(image):
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #image = cv2.equalizeHist(image)
    image = image/255
    #image = transform_image(image,20,3,20)

    return image

def LR_preprocessor(ID):
    images = []

    vidcap = cv2.VideoCapture(ID)
    success, image = vidcap.read()
    success = True
    while success:
        image = lip_reading_image_processing(image)
        images.append(image)
        success, image = vidcap.read()

    # big_window = []
    # for i in range(len(images)-4):
    #     small_window = []
    #     for j in range(5):
    #         small_window.append((images[i+j]))
    #     big_window.append(small_window)

    return images



def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    # print(random_bright)
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def transform_image(img, ang_range, shear_range, trans_range, brightness=0):

    # Rotation

    ang_rot = np.random.uniform(ang_range) - ang_range / 2

    rows, cols = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

    # Brightness

    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

    shear_M = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img, Rot_M, (cols, rows))
    img = cv2.warpAffine(img, Trans_M, (cols, rows))
    img = cv2.warpAffine(img, shear_M, (cols, rows))

    if brightness == 1:
        img = augment_brightness_camera_images(img)

    return img

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_path, batch_size=32, shuffle=True, augment=True,output_shape=(26,120,120,5)):
        self.output_shape = output_shape
        self.solver = LR_preprocessor
        self.data_path = data_path
        self.path = data_path
        self.n_classes = len(os.listdir(self.path))

        self.classes = {}
        for i,classname in enumerate(os.listdir(self.path)):
            self.classes.update({classname:i})

        self.labels = {}
        self.list_IDs = []

        for classname in os.listdir(self.path):
            class_path = os.path.join(self.path,classname)
            for one_data in os.listdir(class_path):
                one_data_path = os.path.join(class_path,one_data)
                self.list_IDs.append(one_data_path)
                self.labels.update({one_data_path:self.classes[classname]})

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        x, y = self.__data_generation(list_IDs_temp)
        return x, y

    def __data_generation(self, list_IDs_temp):
        data = []

        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            data.append(self.solver(ID))
            y[i] = self.labels[ID]

        data = np.array(data)

        data = data.reshape(self.batch_size,*self.output_shape)

        return data, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
    def get_classes(self):
        return self.classes


def identity_block(X, f, filters, stage, block):
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape=(120,120,3)):
    # Define the input as a tensor with shape input_shape
    X_input = Input(shape=input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    # Output layer
    #X = Flatten()(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model





class LR_resnet(tf.keras.Model):
    def __init__(self,num_classes):
        super(LR_resnet, self).__init__()

        self.resnet1 = layers.TimeDistributed(ResNet50(input_shape=(120,120,3)))
        self.flat1 = layers.TimeDistributed(Flatten())

        self.fc2 = layers.TimeDistributed(Dense(4096))

        self.bilstm3 = layers.Bidirectional(layers.LSTM(512, return_sequences=True))
        self.dropout3 = layers.Dropout(0.5)

        self.bilstm4 = layers.Bidirectional(layers.LSTM(512))
        self.dropout4 = layers.Dropout(0.5)

        self.fc5 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):

        x = self.resnet1(inputs)
        x = self.flat1(x)

        x = self.fc2(x)

        x = self.bilstm3(x)
        x = self.dropout3(x)

        x = self.bilstm4(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        return x













