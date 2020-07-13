import tensorflow as tf
import os
from internel.nn.model import Lip_reading
from internel.nn import Resnet_generator
from environment import ROOT_DIR,MODEL_DIR
from tensorflow.keras.callbacks import EarlyStopping
import pickle
class Network:
    def __init__(self, name='[2CLASS]_[TD]_[RES]_[BILSTM]_[GEN5]',batch_size=4):

        self.batch_size = batch_size
        self.callbacks = []

    def init_generator(self):
        train_data_path1 = os.path.join(ROOT_DIR, 'data', 'dataset', 'face', 'testset_light_source1')
        self.train_generator1 = Resnet_generator(data_path=train_data_path1,
                                batch_size=self.batch_size,
                                output_shape=(30,120,120,1),
                                augment=False)

        train_data_path2 = os.path.join(ROOT_DIR, 'data', 'dataset', 'face', 'testset_light_source2')
        self.train_generator2 = Resnet_generator(data_path=train_data_path2,
                                batch_size=self.batch_size,
                                output_shape=(30,120,120,1),
                                augment=False)

        train_data_path3 = os.path.join(ROOT_DIR, 'data', 'dataset', 'face', 'testset_light_source3')
        self.train_generator3 = Resnet_generator(data_path=train_data_path3,
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
        early_stop_callback = EarlyStopping(monitor='val_loss', verbose=1,patience=10)

        if not os.path.exists(os.path.join(MODEL_DIR, 'resnet')):
            os.makedirs(os.path.join(MODEL_DIR, 'resnet'))
        f = open(os.path.join(MODEL_DIR, 'resnet', 'class_info.pickle'), "wb")
        pickle.dump(self.train_generator1.classes, f)
        f.close()
        for i in range(80):
            if i%3 ==0:
                self.model.fit(x=self.train_generator1,validation_data=self.validation_generator,
                               callbacks=[early_stop_callback])
            elif i % 3 == 1:
                self.model.fit(x=self.train_generator2, validation_data=self.validation_generator,
                               callbacks=[early_stop_callback])
            elif i % 3 == 2:
                self.model.fit(x=self.train_generator3, validation_data=self.validation_generator,
                               callbacks=[early_stop_callback])



        self.model.save(os.path.join(MODEL_DIR,'resnet','model.h5'))
        self.test()

    def test(self):
        results = {}
        self.model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'resnet', 'model.h5'))
        for datasets in os.listdir(os.path.join(ROOT_DIR, 'data', 'dataset', 'face')):
            path = os.path.join(ROOT_DIR, 'data', 'dataset', 'face',datasets)
            self.test_generator = Resnet_generator(data_path=path,
                                                         batch_size=self.batch_size,
                                                         output_shape=(30, 120, 120, 1),
                                                         augment=False)
            result = self.model.evaluate(self.test_generator)
            results.update({datasets:result})

        print(results)


network = Network(name='test_model',batch_size=4)
network.init_generator()

network.build()
network.fit()
#network.test()