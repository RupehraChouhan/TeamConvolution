import tensorflow as tf
import numpy as np


class TextureImages(object):
    def __init__(self, subset='train', batch_size=64, shuffle=True):
        if subset == 'train':
            images = np.load('../train_X.npy')
            labels = np.load('../train_Y.npy')
        elif subset == 'valid':
            images = np.load('../valid_X.npy')
            labels = np.load('../valid_Y.npy')
        elif subset == 'test':
            images = np.load('../test_X.npy')
            labels = np.load('../test_Y.npy')
            
        else:
            raise NotImplementedError
        self._images = images
        self.images = self._images
        self._labels = labels
        self.labels = self._labels
        self.batch_size = batch_size
        self.num_samples = len(self.images)
        self.shuffle = shuffle
        if self.shuffle:
            self.shuffle_samples()
        self.next_batch_pointer = 0

    def shuffle_samples(self):
        image_indices = np.random.permutation(np.arange(self.num_samples))
        self.images = self._images[image_indices]
        self.labels = self._labels[image_indices]

    def get_next_batch(self):
        num_samples_left = self.num_samples - self.next_batch_pointer
        if num_samples_left >= self.batch_size:
            x_batch = self.images[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]
            temp_x = []
            temp_y = []
            for itemx in x_batch:
                temp_x.append( itemx.reshape(64,64,1))
            y_batch = self.labels[self.next_batch_pointer:self.next_batch_pointer + self.batch_size]  
            for itemy in y_batch:
                temp_y.append(np.reshape(itemy,(2)))             
            self.next_batch_pointer += self.batch_size
        else:
            x_partial_batch_1 = self.images[self.next_batch_pointer:self.num_samples]
            y_partial_batch_1 = self.labels[self.next_batch_pointer:self.num_samples]
            if self.shuffle:
                self.shuffle_samples()
            x_partial_batch_2 = self.images[0:self.batch_size - num_samples_left]
            y_partial_batch_2 = self.labels[0:self.batch_size - num_samples_left]
            x_batch = np.vstack((x_partial_batch_1, x_partial_batch_2))
            temp_x = []
            temp_y = []
            for itemx in x_batch:
                temp_x.append(itemx.reshape(64,64,1))            
            y_batch = np.vstack((y_partial_batch_1, y_partial_batch_2))  
            for itemy in y_batch:
                temp_y.append(np.reshape(itemy,(2)))             
            self.next_batch_pointer = self.batch_size - num_samples_left
        return temp_x, temp_y

    def get_full_set(self):
        set_x = self.images
        set_y = self.labels
        temp_x = []
        temp_y = []
        for itemx in set_x:
            temp_x.append(itemx.reshape(64,64,1))
        for itemy in set_y:
            temp_y.append(np.reshape(itemy, (2)))
        return temp_x, temp_y