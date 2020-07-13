import tensorflow as tf
import os
import numpy as np
from natsort import natsorted
from multiprocessing import Pool
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_paths, loader, batch_size,shuffle,transformer = None):
        self.loader = loader
        self.transformer = transformer
        self.data_paths = data_paths
        self.batch_size = batch_size
        self.shuffle = shuffle


        self.category_initializer()
        self.dataset_mapper()

        self.on_epoch_end()

    def category_initializer(self):
        if self.data_paths is None:
            print("[DataGenerator] Datasets are given as None")
            exit(1)
        #### map list of categories from dataset paths
        categories = []

        for _path in self.data_paths:
            temp = {}

            for i,label in enumerate(natsorted(os.listdir(_path))):
                temp.update({label:i})
            categories.append(temp)

        #### check integrity of category data
        for _dictionary in categories:
            if _dictionary == categories[0]:
                pass
            else:
                print("List of dataset MUST have identical composition of caegories ")
                exit(1)

        #### initialize num_classes and labels data
        self.num_classes = len(categories[0])
        self.labels = categories[0]

    def dataset_mapper(self):

        self.mapped_dataset = {}

        for _path_dataset in self.data_paths:
            for _label in list(self.labels.keys()):
                _path_single_class = os.path.join(_path_dataset,_label)
                for _name_file in os.listdir(_path_single_class):
                    _path_single_file = os.path.join(_path_single_class,_name_file)
                    self.mapped_dataset.update({_path_single_file:self.labels[_label]})

        self.iterator = list(self.mapped_dataset.keys())

    def __len__(self):
        return int(np.floor(len(self.iterator) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.iterator))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_iterator = [self.iterator[k] for k in indexes]
        x, y = self.__data_generation(batch_iterator)

        return x, y

    def __data_generation(self, batch_iterator):
        x = []
        y = []


        for _path in batch_iterator:
            data = self.loader(_path)
            if self.transformer is not None:
                data = self.transformer(data, _path)

            x.append(data)
            y.append( self.mapped_dataset[_path])



        x = np.stack( x, axis=0 )


        return x, tf.keras.utils.to_categorical(y, num_classes=self.num_classes)


