import tensorflow as tf
import os
from framework.internel.nn.model import Lip_reading
from framework.internel.nn.generator import Resnet_generator
from framework.environment.definitions import ROOT_DIR,MODEL_DIR
import pickle
class Network:
    def __init__(self, name='[2CLASS]_[TD]_[RES]_[BILSTM]_[GEN5]',batch_size=4):

        self.batch_size = batch_size
        self.callbacks = []

    def init_generator(self):
        train_data_path = os.path.join(ROOT_DIR, 'data', 'dataset', 'face', 'train')
        self.train_generator = Resnet_generator(data_path=train_data_path,
                                batch_size=self.batch_size,
                                output_shape=(30,120,120,1),
                                augment=False)

        validation_data_path = os.path.join(ROOT_DIR, 'data', 'dataset', 'face', 'validation')
        self.validation_generator = Resnet_generator(data_path=validation_data_path,
                                batch_size=self.batch_size,
                                output_shape=(30,120,120,1),
                                augment=False)

    def build(self):
        self.model = Lip_reading(num_classes=2)
        #self.model.build(input_shape=(4,30,120,120,3))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])



    def fit(self,epochs=500):

        #latest = tf.train.latest_checkpoint(self.weight_dir)

        #if latest is not None: CX
        #    self.model.load_weights(latest)
        #    print("Loading latest weight from path: " + str(latest))
        #else:
        #    print("No existing weight. Strat training from scratch")
        if not os.path.exists(os.path.join(MODEL_DIR, 'resnet')):
            os.makedirs(os.path.join(MODEL_DIR, 'resnet'))
        f = open(os.path.join(MODEL_DIR, 'resnet', 'class_info.pickle'), "wb")
        pickle.dump(self.train_generator.classes, f)
        f.close()
        self.model.fit(x=self.train_generator,validation_data=self.validation_generator, epochs=20)
        self.model.save(os.path.join(MODEL_DIR,'resnet','model.h5'))


network = Network(name='test_model',batch_size=4)
network.init_generator()

network.build()
network.fit()
