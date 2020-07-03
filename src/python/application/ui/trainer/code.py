import PIL
from PIL import Image,ImageTk
import cv2
from tkinter import *
import time
import os
from definitions import ROOT_DIR,MODEL_DIR,INPUT_SHAPE
import tkinter as tk
import uuid
import numpy as np
import dlib
import src.python.application.ui as Ui
from src.python.internel.function import crop_mouth,save_video,load_configuration

class Trainer:
    def __init__(self):
        self.init_window()
        self.init_variables()
        self.init_components()
        self.init_directories()
        self.run()

    def init_window(self):
        self.window = Tk()
        self.window.protocol("WM_DELETE_WINDOW", self.event_close)

    def init_variables(self):
        self.configuration = load_configuration()
        self.input_shape = INPUT_SHAPE[self.configuration['Model']]
        print(self.input_shape)
    def init_components(self):
        pass

    def init_directories(self):
        pass

    def event_close(self):
        self.window.destroy()
        Ui.Main()
        del self

    def ui_start_command(self):
        self.switch = True

    def run(self):
        self.window.mainloop()