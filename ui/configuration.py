from tkinter import *
import os
from environment.variable import CONFIG_PATH,DATASET_DIR,MODEL_LIST
from tkinter import ttk
import pickle
import shutil
import ui as Ui

class Configuration:
    def __init__(self):

        self.init_window()
        self.load_configuration()
        self.init_components()
        self.run()

    def load_configuration(self):
        self.configuration = pickle.load(open(CONFIG_PATH, "rb"))


    def init_window(self):
        self.window = Tk()
        self.window.protocol("WM_DELETE_WINDOW", self.event_close)

    def init_components(self):

        self.notebook = ttk.Notebook(self.window)
        self.init_record_tab()
        self.init_train_tab()

        self.notebook.grid(row=0)


        self.ui_save = Ui.LABEL_WITH_BUTTON(self.window,
                                            '',
                                            '[SAVE]',
                                            self.ui_save_command)
        self.ui_save.frame.grid(row=1, padx=20, pady=5)


    def init_record_tab(self):
        self.record_setting_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.record_setting_tab, text='Record')

        self.ui_name_dataset = Ui.LABEL_WITH_ENTRY(self.record_setting_tab,
                                                    'Name Dataset',
                                                    str(self.configuration['Name Dataset']),
                                                    20)
        self.ui_name_dataset.frame.grid(row=0, padx=20, pady=5)

        self.ui_name_class = Ui.LABEL_WITH_ENTRY(self.record_setting_tab,
                                                    'Name Class',
                                                    str(self.configuration['Name Class']),
                                                    20)
        self.ui_name_class.frame.grid(row=1, padx=20, pady=5)


        self.ui_sampling_rate = Ui.LABEL_WITH_ENTRY(self.record_setting_tab,
                                                    'Sampling Rate',
                                                    str(self.configuration['Sampling Rate']),
                                                    20)
        self.ui_sampling_rate.frame.grid(row=2, padx=20, pady=5)

        self.ui_camera_width = Ui.LABEL_WITH_ENTRY(self.record_setting_tab,
                                                   'Camera Width',
                                                   str(self.configuration['Camera Width']),
                                                   20)
        self.ui_camera_width.frame.grid(row=3, padx=20, pady=5)

        self.ui_camera_height = Ui.LABEL_WITH_ENTRY(self.record_setting_tab,
                                                    'Camera Height',
                                                    str(self.configuration['Camera Height']),
                                                    20)
        self.ui_camera_height.frame.grid(row=4, padx=20, pady=5)

        self.ui_frames_in_one_sample = Ui.LABEL_WITH_ENTRY(self.record_setting_tab,
                                                           'Frames In One Sample',
                                                           str(self.configuration['Frames In One Sample']),
                                                           20)
        self.ui_frames_in_one_sample.frame.grid(row=5, padx=20, pady=5)

        self.ui_save_original = Ui.LABEL_WITH_BUTTON(self.record_setting_tab,
                                                     'Save Original',
                                                     str(self.configuration['Save Original']),
                                                     self.ui_save_original_command)
        self.ui_save_original.frame.grid(row=6, padx=20, pady=5)

        self.ui_delete_all_dataset = Ui.LABEL_WITH_BUTTON(self.record_setting_tab,
                                                          'DELETE ALL DATASET',
                                                          'DELETE',
                                                          self.ui_delete_all_dataset_command)
        self.ui_delete_all_dataset.frame.grid(row=7, padx=20, pady=5)



    def init_train_tab(self):
        self.train_setting_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_setting_tab, text='Train')

        self.ui_batch_size = Ui.LABEL_WITH_ENTRY(self.train_setting_tab,
                                                    'Batch Size',
                                                    str(self.configuration['Batch Size']),
                                                    20)
        self.ui_batch_size.frame.grid(row=0, padx=20, pady=5)

        self.ui_model_option_dropdown = Ui.LABEL_WITH_DROPDOWN(self.train_setting_tab,
                                                               'Model Option',
                                                               str(self.configuration['Model']),
                                                                MODEL_LIST)
        self.ui_model_option_dropdown.frame.grid(row=1,padx=70,pady=10)

    def ui_save_original_command(self):
        if self.ui_save_original.button.config('text')[-1] == 'True':
            self.ui_save_original.button.config(text='False')
        else:
            self.ui_save_original.button.config(text='True')

    def ui_delete_all_dataset_command(self):
        if os.path.exists(DATASET_DIR):
            shutil.rmtree(DATASET_DIR)

    def ui_save_command(self):

        config = {'Name Dataset' : str(self.ui_name_dataset.entry.get()),
                  'Name Class' : str(self.ui_name_class.entry.get()),
                  'Sampling Rate': int(self.ui_sampling_rate.entry.get()),
                  'Camera Width': int(self.ui_camera_width.entry.get()),
                  'Camera Height': int(self.ui_camera_height.entry.get()),
                  'Frames In One Sample': int(self.ui_frames_in_one_sample.entry.get()),
                  'Batch Size': int(self.ui_batch_size.entry.get()),
                  'Model':str(self.ui_model_option_dropdown.string_var.get())
                  }
        if  self.ui_save_original.button.config('text')[-1] == 'True':
            config.update({'Save Original':True})
        else:
            config.update({'Save Original': False})

        f = open(CONFIG_PATH, "wb")
        pickle.dump(config, f)
        f.close()

        self.event_close()


    def event_close(self):
        self.window.destroy()
        Ui.Main()
        del self

    def run(self):
        self.window.mainloop()

