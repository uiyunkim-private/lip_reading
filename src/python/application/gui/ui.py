import PIL
from PIL import Image,ImageTk
import cv2
from tkinter import *
import time
import json
import os
import csv
from definitions import ROOT_DIR,CONFIG_PATH
import tkinter as tk
from tkinter import ttk
import uuid
from imutils import face_utils
import numpy as np
import dlib
from multiprocessing import Process,Queue
import multiprocessing
import threading
import pickle

class Recorder:
    def __init__(self,window,camera_weight,camera_height):
        self.init_window(window)
        self.init_variables()
        self.init_camera()

        self.init_components()

        self.run()

    def init_window(self,window):
        self.window = window
        self.window.protocol("WM_DELETE_WINDOW", self.event_close)

    def init_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def init_variables(self):

        configuration = pickle.load(open(CONFIG_PATH, "rb"))

        self.camera_width = int(configuration['Camera Width'])
        self.camera_height = int(configuration['Camera Height'])
        self.frames_in_one_sample = int(configuration['Frames In One Sample'])
        self.save_original = bool(configuration['Save Original'])

        self.switch = False
        self.stopwatch = self.frames_in_one_sample
        self.tick = time.time()
        self.framerate_arr = []
        self.framecount = 0
        self.sampling_rate = int(configuration['Sampling Rate'])
        self.framerate_text = tk.StringVar()
        self.info_text = tk.StringVar()
        self.info_error_text= tk.StringVar()
        self.recorded_frames=[]
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor(os.path.join(ROOT_DIR,"system","shape_predictor_68_face_landmarks.dat"))
        self.face_shape =(120,120)

    def init_components(self):
        self.framerate_label = Label(self.window,textvariable=self.framerate_text)
        self.framerate_label.pack()


        self.frame = Label(self.window)
        self.frame.pack()

        self.info_label = Label(self.window,textvariable=self.info_text)
        self.info_label.pack()

        self.info_error_label = Label(self.window,textvariable=self.info_error_text)
        self.info_error_label.pack()

        self.start_button = Button(self.window, text="Record", command=self.start)
        self.start_button.pack()

    def event_close(self):
        cv2.destroyAllWindows()
        self.frame.after_cancel(self.frame_after_id)
        self.cap.release()
        self.window.destroy()
        self.window = Tk()
        Main(self.window)
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
            if self.framecount % self.sampling_rate == 0:
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
                    self.save_video()
                    self.switch = False
                    self.stopwatch = 30

            else:
                self.info_text.set("Press Record To Start Recording")
            self.frame_after_id = self.frame.after(1, self.record)
            self.framecount = self.framecount + 1

    def crop_mouth(self, image):
        rects = self.face_detector(image, 0)
        if len(rects) != 1:
            return None

        shape = self.face_predictor(image, rects[0])
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = cv2.boundingRect(np.array([shape[48:68]]))
        ratio = 70 / w

        image = cv2.resize(image,dsize=(0,0),fx=ratio,fy=ratio)
        x = x * ratio
        y = y * ratio
        w = w * ratio
        h = h * ratio
        midy = y + h / 2
        midx = x + w / 2
        xshape = self.face_shape[1] / 2
        yshape = self.face_shape[0] / 2
        mouth_image = image[int(midy - yshape):int(midy + yshape), int(midx - xshape):int(midx + xshape)]
        return mouth_image

    def save_video(self):
        image_count = 0
        random_filename = str(uuid.uuid4())
        faces = []
        for image in self.recorded_frames:
            face = self.crop_mouth(image)

            if face is None:
                self.info_error_text.set("[Error] No Face Detected On Some Frame.")
                self.recorded_frames = []
                return
            faces.append(np.copy(face))

            cv2.imshow('Faces', face)
            cv2.waitKey(1) & 0xFF
            image_count = image_count + 1
        self.recorded_frames = []

    def start(self):
        self.switch = True

    def run(self):
        self.record()
        self.window.mainloop()




class Main:
    def __init__(self,window):
        self.create_config_file()
        self.init_window(window)
        self.init_components()
        self.run()

    def init_window(self,window):
        self.window = window
        self.window.protocol("WM_DELETE_WINDOW", self.event_close)

    def create_config_file(self):
        if not os.path.exists(CONFIG_PATH):
            config ={'Sampling Rate':30,'Camera Width':1280 , 'Camera Height':720, 'Frames In One Sample':30}
            f = open(CONFIG_PATH, "wb")
            pickle.dump(config, f)
            f.close()

    def init_components(self):
        self.record_button = Button(self.window, text="Record Dataset", command=self.open_recorder)
        self.record_button.pack()

        self.configuration_button = Button(self.window, text="Configuration", command=self.open_configuration)
        self.configuration_button.pack()

    def event_close(self):
        self.window.destroy()
        self.window.quit()
        del self

    def open_configuration(self):
        self.window.destroy()
        self.window = Tk()
        Configuration(self.window)
        del self

    def open_recorder(self):
        self.window.destroy()
        self.window = Tk()
        Recorder(self.window,camera_weight=1280,camera_height=720)
        del self

    def run(self):
        self.window.mainloop()

class Configuration:
    def __init__(self,window):
        self.init_window(window)
        self.load_current_config()
        self.init_components()
        self.run()

    def load_current_config(self):
        configuration = pickle.load( open(CONFIG_PATH, "rb" ) )

        self.sampling_rate = int(configuration['Sampling Rate'])
        self.camera_width = int(configuration['Camera Width'])
        self.camera_height = int(configuration['Camera Height'])
        self.frames_in_one_sample = int(configuration['Frames In One Sample'])



    def init_window(self,window):
        self.window = window
        self.window.protocol("WM_DELETE_WINDOW", self.event_close)

    def init_components(self):
        self.notebook = ttk.Notebook(self.window)

        self.camera_setting_tab = ttk.Frame(self.notebook, width = 400, height = 400)
        self.trainer_setting_tab = ttk.Frame(self.notebook, width = 400, height = 400)

        self.notebook.add(self.camera_setting_tab, text = 'Camera')
        self.notebook.add(self.trainer_setting_tab, text = 'Trainer')

        self.sampling_rate_label = Label(self.camera_setting_tab,text="Sampling Rate")
        self.sampling_rate_label.pack()
        self.sampling_rate_scale = Scale(self.camera_setting_tab, from_=1, to=30, orient=HORIZONTAL)
        self.sampling_rate_scale.pack()
        self.sampling_rate_scale.set(self.sampling_rate)

        self.blank1 = Label(self.camera_setting_tab,text="------------------------------------------")
        self.blank1.pack()

        self.frames_in_one_sample_label = Label(self.camera_setting_tab,text="Frames In One Sample")
        self.frames_in_one_sample_label.pack()
        self.frames_in_one_sample_scale = Scale(self.camera_setting_tab, from_=5, to=30, orient=HORIZONTAL)
        self.frames_in_one_sample_scale.pack()
        self.frames_in_one_sample_scale.set(self.frames_in_one_sample)

        self.blank2 = Label(self.camera_setting_tab,text="------------------------------------------")
        self.blank2.pack()

        self.camera_width_label = Label(self.camera_setting_tab,text="Camera Width")
        self.camera_width_label.pack()
        self.camera_width_scale = Scale(self.camera_setting_tab, from_=640, to=1920, orient=HORIZONTAL)
        self.camera_width_scale.pack()
        self.camera_width_scale.set(self.camera_width)

        self.blank3 = Label(self.camera_setting_tab,text="------------------------------------------")
        self.blank3.pack()

        self.camera_height_label = Label(self.camera_setting_tab,text="Camera Height")
        self.camera_height_label.pack()
        self.camera_height_scale = Scale(self.camera_setting_tab, from_=480, to=1080, orient=HORIZONTAL)
        self.camera_height_scale.pack()
        self.camera_height_scale.set(self.camera_height)


        self.blank4 = Label(self.camera_setting_tab,text="------------------------------------------")
        self.blank4.pack()

        self.save_original_label = Label(self.camera_setting_tab,text="Save Original")
        self.save_original_label.pack()
        self.save_original_button = tk.Button(self.camera_setting_tab, text="True", width=12, command=self.save_original_toggle)
        self.save_original_button.pack()


        self.notebook.pack()

        self.save_button = Button(self.window, text="Save", command=self.save_configuration)
        self.save_button.pack()
    def save_original_toggle(self):
        if self.save_original_button .config('text')[-1] == 'True':
            self.save_original_button .config(text='False')
        else:
            self.save_original_button .config(text='True')

    def save_configuration(self):
        config = {'Sampling Rate': self.sampling_rate_scale.get(),
                  'Camera Width': self.camera_width_scale.get(),
                  'Camera Height': self.camera_height_scale.get(),
                  'Frames In One Sample': self.frames_in_one_sample_scale.get(),
                  'Save Original': self.save_original_button.config('text')[-1]}



        f = open(CONFIG_PATH, "wb")
        pickle.dump(config, f)
        f.close()


    def event_close(self):
        self.window.destroy()
        self.window = Tk()
        Main(self.window)
        del self

    def run(self):
        self.window.mainloop()
if __name__ == "__main__":
    Main(Tk())