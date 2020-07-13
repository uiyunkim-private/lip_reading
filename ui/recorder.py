import PIL
from PIL import Image,ImageTk
import cv2
from tkinter import *
import time
from environment.variable import CONFIG_PATH
import tkinter as tk
import pickle
import ui as Ui
from internel.protocol import SampleWriter
class Recorder:
    def __init__(self):
        self.init_window()
        self.load_configuration()
        self.init_variables()
        self.init_components()
        self.init_camera()
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

    def load_configuration(self):
        self.configuration = pickle.load(open(CONFIG_PATH, "rb"))

    def init_variables(self):
        self.stopwatch = self.configuration['Frames In One Sample']

        self.name_dataset = self.configuration['Name Dataset']
        self.name_class = self.configuration['Name Class']

        self.switch = False
        self.tick = time.time()
        self.framerate_arr = []
        self.framecount = 0
        self.framerate_text = tk.StringVar()
        self.info_text = tk.StringVar()
        self.info_error_text= tk.StringVar()
        self.recorded_frames=[]

        self.face_shape =(120,120)
        self.sample_writer = SampleWriter(self.face_shape,self.name_dataset,self.name_class,self.configuration['Frames In One Sample'])

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
                    self.sample_writer.queue.put(self.recorded_frames)
                    self.recorded_frames = []
                    self.switch = False
                    self.stopwatch = self.configuration['Frames In One Sample']

            else:
                self.info_text.set("Press Record To Start Recording")
            self.frame_after_id = self.frame.after(1, self.record)
            self.framecount = self.framecount + 1




    def ui_start_command(self):
        self.switch = True

    def run(self):
        self.record()
        self.window.mainloop()

