from tkinter import *
import tkinter as tk
from tkinter import ttk

class LABEL_WITH_ENTRY:
    def __init__(self,window,label,entry_label,width):
        self.window = window
        self.label = label
        self.entry_label = entry_label
        self.width = width
        self.create()

    def create(self):
        self.frame = ttk.Frame(self.window)
        self.label = Label(self.frame, text=self.label).grid(row=0)
        self.entry = ttk.Entry(self.frame, width=self.width, textvariable=str)
        self.entry.grid(row=0,column=1)
        self.entry.insert(0,self.entry_label)

    def pack(self):
        self.frame.pack()

class LABEL_WITH_BUTTON:
    def __init__(self,window,label,button_label,command):
        self.window = window
        self.label = label
        self.button_label = button_label
        self.command = command
        self.create()

    def create(self):
        self.frame = ttk.Frame(self.window)
        self.label = Label(self.frame, text=self.label).grid(row=0)
        self.button = tk.Button(self.frame,
                                text=self.button_label,
                                width=12,
                                command=self.command)

        self.button.grid(row=0,column=1)

    def pack(self):
        self.frame.pack()



