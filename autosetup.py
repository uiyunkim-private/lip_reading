import os
import sys


#sys.path.append(os.path.dirname(os.path.realpath(__file__)))
os.system("conda create -n lip_reading python=3.7")
os.system("activate lip_reading")
os.system("conda install -y cmake=3.17.0")
os.system("conda install -c conda-forge -y dlib=19.19")
os.system("conda install -c conda-forge -y imutils=0.5.3")
os.system("conda install -y tensorflow-gpu")
