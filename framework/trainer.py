import tensorflow as tf
import os
from framework import DataGenerator
from environment.variable import MODEL_DIR
import pickle

class ModelTrainer:
    def __init__(self,model,train_datasets,data_loader,validation_datasets=None, test_datasets = None,name='default',batch_size=4,data_transformer=None,shuffle=True,callbacks=None):
        self.name = name
        self.model = model
        self.train_datasets = train_datasets
        self.validation_datasets = validation_datasets
        self.test_datasets = test_datasets
        self.data_loader = data_loader
        self.data_transformer = data_transformer

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.model_fit = False

        self.init_generator()
        self.create_directory()
        self.save_label_data()


    def init_generator(self):
        if self.train_datasets is not None:
            self.train_generator = DataGenerator(data_paths=self.train_datasets,loader=self.data_loader,
                                                 transformer=self.data_transformer,batch_size=self.batch_size,
                                                 shuffle=self.shuffle)
        else:
            self.train_generator = None

        if self.validation_datasets is not None:
            self.validation_generator = DataGenerator(data_paths=self.validation_datasets, loader=self.data_loader,
                                                 transformer=self.data_transformer, batch_size=self.batch_size,
                                                 shuffle=False)
        else:
            self.validation_generator = None


    def create_directory(self):
        if not os.path.exists(os.path.join(MODEL_DIR, self.name)):
            os.makedirs(os.path.join(MODEL_DIR, self.name))

    def save_label_data(self):
        f = open(os.path.join(MODEL_DIR, self.name, 'label_data.pickle'), "wb")
        pickle.dump(self.train_generator.labels , f)
        f.close()

    def build(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])

    def fit(self):
        if self.train_generator is None:
            print("[Trainer] Train dataset is not given. But trying to fit model.")
            exit(1)

        if self.validation_generator is None:
            hist = self.model.fit(x=self.train_generator, callbacks=self.callbacks)
        else:
            hist = self.model.fit(x=self.train_generator, validation_data=self.validation_generator,callbacks=self.callbacks)

        self.model_fit = True

        return hist

    def save_model_data(self):
        if self.model_fit:
            self.model.save(os.path.join(MODEL_DIR, self.name, 'model.h5'))
        else:
            print("[Trainer] Model not trained. But trying to save model into file.")
            exit(1)


    def test(self):
        results = {}

        if self.test_datasets is not None:
            self.model = tf.keras.models.load_model(os.path.join(MODEL_DIR, self.name, 'model.h5'))

            for dataset in self.test_datasets:
                test_generator = DataGenerator(data_paths=[dataset], loader=self.data_loader,
                                                     transformer=self.data_transformer, batch_size=self.batch_size,
                                                     shuffle=False)

                result = self.model.evaluate(test_generator)
                results.update({dataset:result})
        else:
            print("[Trainer] Test dataset is not given. But trying to test model")
            exit(1)



        return results

