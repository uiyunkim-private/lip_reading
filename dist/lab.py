from lip_reading.network.tf_based.model import BI_LSTM
from lip_reading.network.tf_based.util import LR_generator
from lip_reading.network.tf_based.util import LoadLatestWeight
import os
import tensorflow as tf
import sys
import pickle
import time
import cv2
def dataset_check():
    input_shape = (26, 120, 120, 5)
    batch_size = 4
    module_path = sys.path[1]
    train_data_path = os.path.join(module_path, 'lip_reading', 'storage', 'dataset', 'cut')
    train_generator = LR_generator(batch_size=batch_size, data_path=train_data_path, type='train', shape=input_shape)

    for i in range(len(train_generator)):
        sample = train_generator[i][0]
        print(sample.shape)
        for j in range(4):
            each_batch = sample[j]
            print(each_batch.shape)
            for k in range(26):
                small_window = each_batch[k]
                print(small_window.shape)
                small_window = small_window.reshape(small_window.shape[2],small_window.shape[0],small_window.shape[1])
                print(small_window.shape)
                for l in range(5):
                    image = small_window[l]
                    print(image)
                    time.sleep(0.3)

    #cv2.destroyAllWindows()

dataset_check()