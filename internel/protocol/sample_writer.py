from imutils import face_utils
import cv2
import numpy as np
import dlib
import os
from environment.variable import ROOT_DIR,DATASET_DIR
from multiprocessing import Process,Queue
import uuid

class SampleWriter():
    def __init__(self,face_shape,name_dataset,name_class,fps):
        self.name_dataset = name_dataset
        self.name_class = name_class

        self.face_detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor(os.path.join(ROOT_DIR, "environment", "shape_predictor_68_face_landmarks.dat"))

        self.count = 0
        self.face_shape = face_shape
        self.fps = fps

        self.switch = True

        self.init_directories()

        self.queue = Queue()
        self.process = Process(target=self.main)
        self.process.start()

    def init_directories(self):
        if not os.path.exists(os.path.join(DATASET_DIR,'face',self.name_dataset,self.name_class)):
            os.makedirs(os.path.join(DATASET_DIR,'face',self.name_dataset,self.name_class))

        if not os.path.exists(os.path.join(DATASET_DIR,'original',self.name_dataset,self.name_class)):
            os.makedirs(os.path.join(DATASET_DIR,'original',self.name_dataset,self.name_class))

    def off(self):
        self.switch = False
        self.process.join()

    def main(self):
        while True:
            if not self.queue.empty():
                self.data = self.queue.get()
                self.pipeline()
                print('saved.')
                print('remained queue: ',self.queue.qsize())
            else:
                if not self.switch:
                    return

    def pipeline(self):
        self.video_saver('original')
        self.face_cropper()
        self.video_saver('face')


    def video_saver(self,definition):
        if self.data is None:
            return

        random_filename = str(uuid.uuid4())  + '.mp4'
        path = os.path.join(DATASET_DIR,definition,self.name_dataset,self.name_class,random_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print((self.data[0].shape[0], self.data[0].shape[1]))
        out = cv2.VideoWriter(path, fourcc, self.fps, (self.data[0].shape[1], self.data[0].shape[0]))
        for frame in self.data:
            out.write(frame)
        out.release()

    def face_cropper(self):
        if self.data is None:
            return

        outputs = []
        for image in self.data:

            rects = self.face_detector(image, 0)
            if len(rects) != 1:
                self.data = None
                return

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
            xshape = self.face_shape[1] / 2
            yshape = self.face_shape[0] / 2

            cropped = image[int(midy - yshape):int(midy + yshape), int(midx - xshape):int(midx + xshape)]

            outputs.append(cropped)

        self.data = outputs
