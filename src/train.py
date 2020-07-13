from framework import DataGenerator
from framework import ModelTrainer
import cv2
from src.model import Gen1
from src.model.gen1 import resnet,conv1_layer,conv2_layer,conv3_layer,conv4_layer,conv5_layer
import tensorflow as tf
from environment.variable import DATASET_DIR
from matplotlib import pyplot
from imutils import face_utils
import cv2
import numpy as np
import dlib
import os
from environment.variable import ROOT_DIR
import math

class LipReadingLoader:
    def __call__(self,path):

        images = []

        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        success = True
        while success:
            images.append(image)
            success, image = vidcap.read()

        return images

count = 0
class LipReadingTransformer:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor(
            os.path.join(ROOT_DIR, "environment", "shape_predictor_68_face_landmarks.dat"))\

        self.count = 0

    def __call__(self,data,path):
        images = []
        for index in range(len(data)):

            data[index] = self.crop_mouth(data[index],(500,500))



            # if data[index] is None:
            #     os.remove(path)
            #     print(path)
            #     return np.zeros((30,60,80,1))
            #else:
            #data[index] = cv2.cvtColor(data[index], cv2.COLOR_BGR2GRAY)
            #data[index] = cv2.equalizeHist(data[index])
            #cv2.imwrite(os.path.join(ROOT_DIR,'images','face'+str(self.count)+'.png'),data[index])
            # print(self.count)
            # self.count = self.count + 1
            # cv2.imshow(path,data[index])
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            images.append(data[index])



       # print(images)

        if np.array(images).shape != (30, 500, 500, 3):
            return np.zeros((30, 500, 500, 3))

        for index in range(len(data)):
            cv2.imwrite(os.path.join(ROOT_DIR, 'images', 'face' + str(self.count) + '.png'), images[index])
            self.count = self.count + 1


        images = np.array(images)
        images = images.reshape(30,500,500,3)
        #print(images.shape)

        return images

    def crop_mouth(self,image, face_shape):

        rects = self.face_detector(image, 0)
        if len(rects) != 1:
            return None

        shape = self.face_predictor(image, rects[0])
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]]))
        ratio = 70 / w

        image = cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio)
        x = x * ratio
        y = y * ratio
        w = w * ratio
        h = h * ratio
        midy = y + h / 2
        midx = x + w / 2
        xshape = face_shape[1] / 2
        yshape = face_shape[0] / 2

        mouth_image = image[int(midy - yshape):int(midy + yshape), int(midx - xshape):int(midx + xshape)]
        return mouth_image
def VisConv():

    train_dataset = os.path.join(DATASET_DIR,'face','train')
    generator = DataGenerator(data_paths=[train_dataset],
                              loader=LipReadingLoader(),
                              transformer=LipReadingTransformer(),
                              batch_size=1,
                              shuffle=True)

    img = generator[0][0][0][0]
    img = np.expand_dims(img, axis=0)


    model = resnet()
    conv_indexes = []
    for i in range(len(model.layers)):
        layer = model.layers[i]
        # check for convolutional layer
        if 'conv' not in layer.name:
            continue
        # summarize output shape
        conv_indexes.append(i)
        print(i, layer.name, layer.output.shape)

    for index in reversed(conv_indexes):
        model = resnet()
        model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[index].output)
        feature_maps = model.predict(img)
        print(feature_maps.shape)
        # plot all 64 maps in an 8x8 squares
        square = int(math.sqrt(feature_maps.shape[-1]))
        ix = 1
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = pyplot.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                pyplot.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
                ix += 1
        # show the figure
        pyplot.show()



def Train():
    train_dataset = os.path.join(DATASET_DIR,'face','train')
    validation_dataset = os.path.join(DATASET_DIR,'face','validation')

    light_source_1_dataset = os.path.join(DATASET_DIR,'face','testset_light_source1')
    light_source_2_dataset = os.path.join(DATASET_DIR,'face','testset_light_source2')
    light_source_3_dataset = os.path.join(DATASET_DIR,'face','testset_light_source3')

    trainer = ModelTrainer(model=Gen1(num_classes=2),
                           train_datasets=[train_dataset],
                           data_loader=LipReadingLoader(),
                           data_transformer=LipReadingTransformer(),
                           validation_datasets=[validation_dataset],
                           test_datasets=[light_source_1_dataset,light_source_2_dataset,light_source_3_dataset],
                           name='test',batch_size=4,shuffle=True,callbacks=None)


    trainer.build()
    epoch_count = 1



    while True:
        print("epoch: " +str(epoch_count))
        hist = trainer.fit()
        print(hist.history['loss'][0])
        if hist.history['loss'][0] < 0.05:
            break
        epoch_count = epoch_count + 1
    trainer.save_model_data()
    print(trainer.test())
def test_generator():
    # train_dataset = os.path.join(DATASET_DIR,'full','train')
    # validation_dataset = os.path.join(DATASET_DIR,'full','validation')
    # light_source_1_dataset = os.path.join(DATASET_DIR,'full','testset_light_source1')
    # light_source_2_dataset = os.path.join(DATASET_DIR,'full','testset_light_source2')
    light_source_3_dataset = os.path.join(DATASET_DIR,'full','testset_light_source3')

    generator = DataGenerator(data_paths=[light_source_3_dataset],loader=LipReadingLoader(),batch_size=4,shuffle=True,transformer=LipReadingTransformer())

    for i in range(len(generator)):
        generator[i]
if __name__ == '__main__':
    test_generator()