from tkinter import *
from framework.environment.definitions import INPUT_SHAPE
import framework.ui as Ui
from framework.internel.function import load_configuration

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
        self.ui_start_training = Ui.LABEL_WITH_BUTTON(self.window,
                                                  '',
                                                  '[Start Training]',
                                                  self.ui_start_training_command)
        self.ui_start_training.frame.grid(row=0,padx=70,pady=10)
        pass

    def ui_start_training_command(self):
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
