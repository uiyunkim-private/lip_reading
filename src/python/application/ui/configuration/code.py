from tkinter import *
import os
from definitions import ROOT_DIR, CONFIG_PATH
from tkinter import ttk
import pickle
import shutil
import src.python.application.ui as Ui
import src.python.internel.function as Function

class Configuration:
    def __init__(self):

        self.init_window()
        self.load_current_config()
        self.init_components()
        self.run()

    def load_current_config(self):
        self.configuration = Function.load_configuration()

    def init_window(self):
        self.window = Tk()
        self.window.protocol("WM_DELETE_WINDOW", self.event_close)

    def init_components(self):

        self.notebook = ttk.Notebook(self.window)
        self.record_setting_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.record_setting_tab, text='Record')

        self.ui_sampling_rate = Ui.LABEL_WITH_ENTRY(self.record_setting_tab,
                                                 'Sampling Rate',
                                                 str(self.configuration['Sampling Rate']),
                                                 20)
        self.ui_sampling_rate.frame.grid(row=0,padx=20,pady=5)

        self.ui_camera_width = Ui.LABEL_WITH_ENTRY(self.record_setting_tab,
                                                'Camera Width',
                                                str(self.configuration['Camera Width']),
                                                20)
        self.ui_camera_width.frame.grid(row=1,padx=20,pady=5)

        self.ui_camera_height = Ui.LABEL_WITH_ENTRY(self.record_setting_tab,
                                                 'Camera Height',
                                                 str(self.configuration['Camera Height']),
                                                 20)
        self.ui_camera_height.frame.grid(row=2,padx=20,pady=5)

        self.ui_frames_in_one_sample = Ui.LABEL_WITH_ENTRY(self.record_setting_tab,
                                                        'Frames In One Sample',
                                                        str(self.configuration['Frames In One Sample']),
                                                        20)
        self.ui_frames_in_one_sample.frame.grid(row=3,padx=20,pady=5)

        self.ui_save_original = Ui.LABEL_WITH_BUTTON(self.record_setting_tab,
                                                  'Save Original',
                                                  str(self.configuration['Save Original']),
                                                  self.ui_save_original_command)
        self.ui_save_original.frame.grid(row=4,padx=20,pady=5)


        self.ui_delete_all_dataset = Ui.LABEL_WITH_BUTTON(self.record_setting_tab,
                                                  'DELETE ALL DATASET',
                                                  'DELETE',
                                                  self.ui_delete_all_dataset_command)
        self.ui_delete_all_dataset.frame.grid(row=5,padx=20,pady=5)

        self.ui_save = Ui.LABEL_WITH_BUTTON(self.record_setting_tab,
                                                  '',
                                                  '[SAVE]',
                                                  self.ui_save_command)
        self.ui_save.frame.grid(row=6,padx=20,pady=5)

        self.notebook.pack()

    def ui_save_original_command(self):
        if self.ui_save_original.button.config('text')[-1] == 'True':
            self.ui_save_original.button.config(text='False')
        else:
            self.ui_save_original.button.config(text='True')

    def ui_delete_all_dataset_command(self):
        if os.path.exists(os.path.join(ROOT_DIR,'data')):
            shutil.rmtree(os.path.join(ROOT_DIR,'data'))

    def ui_save_command(self):
        config = {'Sampling Rate': int(self.ui_sampling_rate.entry.get()),
                  'Camera Width': int(self.ui_camera_width.entry.get()),
                  'Camera Height': int(self.ui_camera_height.entry.get()),
                  'Frames In One Sample': int(self.ui_frames_in_one_sample.entry.get()),
                  }
        if  self.ui_save_original.button.config('text')[-1] == 'True':
            config.update({'Save Original':True})
        else:
            config.update({'Save Original': False})

        f = open(CONFIG_PATH, "wb")
        pickle.dump(config, f)
        f.close()


    def event_close(self):
        self.window.destroy()
        Ui.Main()
        del self

    def run(self):
        self.window.mainloop()

