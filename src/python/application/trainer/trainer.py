import os
import tensorflow as tf
from src.python.base.model import Resnet_Timedistributed_Bilstm
from src.python.base import DataGenerator

class Trainer:
    def __init__(self, name='[2CLASS]_[TD]_[RES]_[BILSTM]_[GEN5]',
                 save_log_tensorboard=True,
                 save_weight=True,
                 save_weight_frequency=5,
                 save_class_dict=True,
                 save_best=True):


        self.name = name
        self.callbacks = []

        if save_log_tensorboard:
            self.log_dir = os.path.join('storage', 'record', 'log', self.name)
            self.callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                                                 update_freq='batch',
                                                                 profile_batch=0))
        if save_weight:
            self.weight_file = os.path.join('storage', 'record', 'weight', self.name,
                                       "cp-{epoch:04d}.ckpt")
            self.weight_dir = os.path.dirname(self.weight_file)
            self.callbacks.append(tf.keras.callbacks.ModelCheckpoint(self.weight_file,
                                                                     verbose=1,
                                                                     save_weights_only=True,
                                                                     period=save_weight_frequency))
        if save_class_dict:
            self.class_dict_dir = os.path.join('storage', 'record', 'configuration', self.name)
        if save_best:
            self.model_file = os.path.join('storage', 'record', 'model', self.name, 'model.h5')

    def build(self):
        print("Batch size:",end=' ')
        self.batch_size = input()

        self.model = Resnet_Timedistributed_Bilstm()
        self.model.compile(optimizer="Adam", loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])

        train_data_path = os.path.join('storage', 'dataset', 'cut')
        self.train_generator = DataGenerator(data_path=train_data_path,
                                batch_size=self.batch_size,
                                output_shape=(30,120,120,3))

    def fit(self,epochs=500):

        latest = tf.train.latest_checkpoint(self.weight_dir)

        if latest is not None:
            self.model.load_weights(latest)
            print("Loading latest weight from path: " + str(latest))
        else:
            print("No existing weight. Strat training from scratch")


        self.model.fit(x=self.train_generator, epochs=1000, callbacks=self.callbacks)