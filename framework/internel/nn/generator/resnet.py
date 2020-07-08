from tensorflow.keras import layers
import tensorflow as tf
import time
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
from tensorflow.keras import Model
import tensorflow as tf
from natsort import natsorted
import cv2
import os
import numpy as np
from framework.environment.definitions import MODEL_DIR,ROOT_DIR
def lip_reading_image_processing(image):
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)



    blurred = cv2.GaussianBlur(image, (33, 33), 0)

    image = image / (blurred ) * 255
    #normalized = cv2.normalize(image,  None, 0, 1, cv2.NORM_MINMAX)

    #print(normalized.shape)
   # image = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)


    #print(normalized)
    #image = 255 - image
    #image = cv2.fastNlMeansDenoising(image, None, 10, 10, 7)
    #image = cv2.equalizeHist(image)

    #image = image/255

    return image

def lip_reading_augmentation(images):

    #images = transform_image(images, 20, 3, 20)
    #images = transform_image(images, 30, 0, 0)
    return images

def LR_preprocessor(ID,augmentation=False):
    images = []

    vidcap = cv2.VideoCapture(ID)
    success, image = vidcap.read()
    success = True
    while success:
        image = lip_reading_image_processing(image)
        images.append(image)
        success, image = vidcap.read()

    if augmentation:
        images = lip_reading_augmentation(images)

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


def transform_image(images, ang_range, shear_range, trans_range, brightness=0):

    # Rotation

    ang_rot = np.random.uniform(ang_range) - ang_range / 2

    rows, cols = images[0].shape
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

    for i in range(len(images)):
        images[i] = cv2.warpAffine(images[i], Rot_M, (cols, rows))
        images[i] = cv2.warpAffine(images[i], Trans_M, (cols, rows))
        images[i] = cv2.warpAffine(images[i], shear_M, (cols, rows))
        #time.sleep(0.05)


    #if brightness == 1:
    #    img = augment_brightness_camera_images(img)

    return images

class Resnet_generator(tf.keras.utils.Sequence):
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
            data.append(self.solver(ID,self.augment))
            y[i] = self.labels[ID]

        data = np.array(data)

        data = data.reshape(self.batch_size,*self.output_shape)

        return data, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
    def get_classes(self):
        return self.classes

def test_generator():
    batch_size = 4
    validation_data_path1 = os.path.join(ROOT_DIR, 'data', 'dataset', 'face', 'testset_light_source2')
    validation_generator1 = Resnet_generator(data_path=validation_data_path1,
                                                 batch_size=batch_size,
                                                 output_shape=(30, 120, 120, 3),
                                                 augment=True)
    validation_data_path2 = os.path.join(ROOT_DIR, 'data', 'dataset', 'face', 'testset_light_source2')
    validation_generator2 = Resnet_generator(data_path=validation_data_path2,
                                                 batch_size=batch_size,
                                                 output_shape=(30, 120, 120, 3),
                                                 augment=True)
    validation_data_path3 = os.path.join(ROOT_DIR, 'data', 'dataset', 'face', 'testset_light_source2')
    validation_generator3 = Resnet_generator(data_path=validation_data_path3,
                                                 batch_size=batch_size,
                                                 output_shape=(30, 120, 120, 3),
                                                 augment=True)

    for q in range(len(validation_generator1)):
        for i in range(batch_size):
            sample = validation_generator1[q][0]
            for j in range(4):
                for k in range(30):

                    print(sample[j][k].shape)
                    print(i,j,k)
                    cv2.imshow('sample',sample[j][k])

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

test_generator()