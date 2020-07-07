import PIL
from PIL import Image,ImageTk
import cv2
from tkinter import *
import time
import os
from framework.environment.definitions import DATASET_DIR,ROOT_DIR,MODEL_DIR
import tkinter as tk
import uuid
import numpy as np
import dlib
import framework.ui as Ui
from framework.internel.function import crop_mouth,save_video,load_configuration
from framework.internel.nn.generator import Resnet_generator
import tensorflow as tf
import pickle
class Tester:
    def __init__(self):
        self.init_window()
        self.init_variables()
        self.init_directories()
        self.init_camera()
        self.init_components()
        self.run()

    def init_window(self):
        self.window = Tk()
        self.window.protocol("WM_DELETE_WINDOW", self.event_close)

    def init_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.configuration['Camera Width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.configuration['Camera Height'])
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def init_variables(self):
        self.model = tf.keras.models.load_model(os.path.join(MODEL_DIR,'resnet','model.h5'))
        self.class_info =  pickle.load(open(os.path.join(MODEL_DIR,'resnet','class_info.pickle'), "rb"))


        self.configuration = load_configuration()
        self.stopwatch = self.configuration['Frames In One Sample']

        self.switch = False
        self.tick = time.time()
        self.framerate_arr = []
        self.framecount = 0
        self.framerate_text = tk.StringVar()
        self.info_text = tk.StringVar()
        self.info_error_text= tk.StringVar()
        self.recorded_frames=[]
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor(os.path.join(ROOT_DIR,'framework',"environment","shape_predictor_68_face_landmarks.dat"))
        self.face_shape =(120,120)

    def init_directories(self):
        if not os.path.exists(os.path.join(DATASET_DIR,'face','test','nolabel')):
            os.makedirs(os.path.join(DATASET_DIR,'face','test','nolabel'))
        if len(os.listdir(os.path.join(DATASET_DIR,'face','test','nolabel'))) != 0:
            for videos in os.listdir(os.path.join(DATASET_DIR,'face','test','nolabel')):
                os.remove(os.path.join(os.path.join(DATASET_DIR,'face','test','nolabel',videos)))



    def init_components(self):
        self.framerate_label = Label(self.window,textvariable=self.framerate_text)
        self.framerate_label.pack()


        self.frame = Label(self.window)
        self.frame.pack()

        self.info_label = Label(self.window,textvariable=self.info_text)
        self.info_label.pack()

        self.info_error_label = Label(self.window,textvariable=self.info_error_text)
        self.info_error_label.pack()


        self.ui_start = Ui.LABEL_WITH_BUTTON(self.window,
                                                  '',
                                                  '[Record Start]',
                                                  self.ui_start_command)
        self.ui_start.pack()



    def event_close(self):
        self.frame.after_cancel(self.frame_after_id)
        self.cap.release()
        cv2.destroyAllWindows()
        self.window.destroy()
        Ui.Main()
        del self

    def show_image(self,image):
        frame = cv2.flip(image, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = PIL.Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.frame.imgtk = imgtk
        self.frame.configure(image=imgtk)

    def record(self):
        ret, frame = self.cap.read()
        if ret:
            now = time.time()
            self.framerate_arr.append(1/(now - self.tick))
            if len(self.framerate_arr) > 30:
                self.framerate_arr.pop(0)
                self.framerate_text.set("FPS: "+ str( int( sum(self.framerate_arr[15:])/15)))

            self.tick = now
            if self.framecount % self.configuration['Sampling Rate'] == 0:
                self.show_image(frame)

            if self.switch:
                self.info_error_text.set("")
                if self.stopwatch > 3:
                    self.info_text.set("Recording: " + str(self.stopwatch-1))
                else:
                    self.info_text.set("Saving To Video")
                self.recorded_frames.append(frame)
                self.stopwatch = self.stopwatch - 1
                if self.stopwatch == 0:
                    self.predict()
                    self.switch = False
                    self.stopwatch = self.configuration['Frames In One Sample']

            else:
                self.info_text.set("Press Record To Start Recording")
            self.frame_after_id = self.frame.after(1, self.record)
            self.framecount = self.framecount + 1


    def predict(self):

        image_count = 0
        random_filename = str(uuid.uuid4())
        faces = []
        for image in self.recorded_frames:
            face = crop_mouth(image,self.face_shape,self.face_detector,self.face_predictor)

            if face is None:
                self.info_error_text.set("[Error] No Face Detected On Some Frame.")
                self.recorded_frames = []
                return
            faces.append(np.copy(face))

            cv2.imshow('Faces', face)
            cv2.waitKey(1) & 0xFF
            image_count = image_count + 1

        path_face = os.path.join(DATASET_DIR,'face','test','nolabel',random_filename + '.mp4')
        save_video(faces,
                   path_face,
                   self.configuration['Frames In One Sample'],
                   self.face_shape[0],
                   self.face_shape[1])

        self.recorded_frames = []


        generator = Resnet_generator(os.path.join(DATASET_DIR,'face','test'), batch_size=1, output_shape=(30, 120, 120, 1),augment=False)

        answer = self.model.predict_classes(generator[0])

        for class_name, index in self.class_info.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
            if index == answer:
                self.info_error_text.set(class_name)
                #print(class_name)
        os.remove(os.path.join(DATASET_DIR,'face','test','nolabel',random_filename + '.mp4'))

    def ui_start_command(self):
        self.switch = True

    def run(self):
        self.record()
        self.window.mainloop()

