from lip_reading.network.tf_based.model import RES_BI_LSTM, VGG_BI_LSTM
from lip_reading.network.tf_based.util import LR_generator
from lip_reading.network.tf_based.util import LoadLatestWeight

import os
import tensorflow as tf
import sys
import pickle

module_path = sys.path[1]
batch_size = 4
input_shape = (30, 120, 120, 3)
model_name = '[2CLASS]_[TD]_[RES]_[BILSTM]_[GEN5]'

log_dir = os.path.join(module_path, 'lip_reading', 'storage', 'record', 'log', model_name)
weight_file = os.path.join(module_path, 'lip_reading', 'storage', 'record', 'weight', model_name, "cp-{epoch:04d}.ckpt")
weight_dir = os.path.dirname(weight_file)
conf_dir = os.path.join(module_path, 'lip_reading', 'storage', 'record', 'configuration', model_name)
model_dir = os.path.join(module_path, 'lip_reading', 'storage', 'record', 'model', model_name,'model.h5')

train_data_path = os.path.join(module_path, 'lip_reading', 'storage', 'dataset', 'cut')
train_generator = LR_generator(batch_size=batch_size, data_path=train_data_path, type='train', shape=input_shape)

os.makedirs(conf_dir, exist_ok=True)
os.makedirs(os.path.dirname(model_dir), exist_ok=True)
with open(os.path.join(conf_dir, 'classes.pkl'), 'wb') as f:
    print(train_generator.get_classes())
    pickle.dump(train_generator.get_classes(), f, pickle.HIGHEST_PROTOCOL)


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch', profile_batch=0)
weightcheckpoint_callback = tf.keras.callbacks.ModelCheckpoint(weight_file, verbose=1, save_weights_only=True, period=15)
#bestmodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_dir, mode='max', monitor='accuracy', verbose=2, save_best_only=True)

callback_list = [tensorboard_callback, weightcheckpoint_callback]#,bestmodel_callback]

model = RES_BI_LSTM(nclasses=2,input_shape = input_shape)

model = LoadLatestWeight(model, weight_dir)
model.fit(x=train_generator, epochs=1000, callbacks=callback_list)

model.save(model_dir)