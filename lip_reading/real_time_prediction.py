import pickle
import os
import sys
import tensorflow as tf
from lip_reading.utils import RealtimePrediction
module_path = sys.path[1]

model_name = 'gen1_RES_TD_BILSTM'
model_dir = os.path.join(module_path, 'lip_reading', 'storage', 'record', 'model', model_name,'model.h5')

conf_dir = os.path.join(module_path, 'lip_reading', 'storage', 'record', 'configuration', model_name)

with open(os.path.join(conf_dir, 'classes.pkl'), 'rb') as f:
    classes = pickle.load(f)

model = tf.keras.models.load_model(model_dir)

RealtimePrediction(model,classes,shape=(120,120))