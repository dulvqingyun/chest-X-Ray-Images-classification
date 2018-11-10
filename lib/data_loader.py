import os
import tensorflow as tf
import numpy as np
import pandas as pd
from lib.config import DataConfig
import scipy.io
import cv2

class DataLoader(object):
    def __init__(self, config, is_train, is_shuffle):
        """
        :param config: input config
        :type config: DataConfig
        :param is_train: is in train phase
        :type is_train: bool
        :param is_shuffle: shuffle data
        :type is_shuffle: bool
        """
        self.config = config
        self.is_shuffle = is_shuffle
        self.is_train = is_train

    def load_data(self):
        raise NotImplementedError

    def generate_batch(self):
        raise NotImplementedError


class ChestXRayDataLoader(DataLoader):
    def __init__(self, config, is_train, is_shuffle):
        super().__init__(config, is_train, is_shuffle)
        self.image_width = self.config.image_width
        self.image_height = self.config.image_height
        self.image_depth = self.config.image_depth

        self.data_dir = self.config.data_dir
        self.val_dir = self.config.val_dir
        self.test_dir = self.config.test_dir
        self.batch_size = self.config.batch_size
        self.n_classes = self.config.n_classes

	
        self.class_names=['NORMAL','PNEUMONIA']
	
        self.input_queue_train = self.load_data_extend(config.data_dir)
        self.input_queue_val = self.load_data_extend(config.val_dir)
        self.input_queue_test = self.load_data_extend(config.test_dir)

 #       self.input_queue = self.load_data()


    def readData(self):
        imgs = []
        labs = []
        for index,name in enumerate(self.class_names):
            class_path=self.test_dir+'/'+name+'/'
            for filename in os.listdir(class_path):
                if filename.endswith('.jpeg'):
                    filename = class_path + filename
        
                    img = cv2.imread(filename)
        
#                    top,bottom,left,right = self.getPaddingSize(img)
                    # 将图片放大， 扩充图片边缘部分
#                    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
                    img = cv2.resize(img, (self.image_height, self.image_width))
                    img = img.astype('float32')/255.0
                    imgs.append(img)
                    labs.append(index)   
         
        imgs = np.array(imgs)
        labs = np.array(labs)           
        return imgs, labs 

    def dense_to_one_hot(self,labels_dense, num_classes):
      """Convert class labels from scalars to one-hot vectors."""
      num_labels = labels_dense.shape[0]
      index_offset = np.arange(num_labels) * num_classes
      labels_one_hot = np.zeros((num_labels, num_classes))
      labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
      return labels_one_hot
 
    def shuffleData(self,data,labels):
        idx = np.arange(data.shape[0])# get a sequence
        np.random.shuffle(idx)
    
        return data[idx, ...],labels[idx],idx 
	
    def load_data_extend(self,path):# only load file names and labels list

        image_list = list()
        label_list = list()
        for index, name in enumerate(self.class_names):
            class_path = path + name + '/'
            for filename in os.listdir(class_path):
                if '.jpeg'  in filename:
                    image_list.append(os.path.join(class_path, filename))
                    label_list.append(index)

        temp = np.array([image_list, label_list])
        temp = temp.transpose()
        np.random.shuffle(temp)
        image_list = list(temp[:, 0])
        label_list = list(temp[:, 1])
        label_list = [int(i) for i in label_list]

        image_list = tf.cast(image_list, tf.string)
        label_list = tf.cast(label_list, tf.int32)

        return tf.train.slice_input_producer([image_list, label_list])


    def generate_batch(self,data_category):
        if data_category=='val':
            label = self.input_queue_val[1]
            image_contents = tf.read_file(self.input_queue_val[0])
        if data_category=='train':
            label = self.input_queue_train[1]
            image_contents = tf.read_file(self.input_queue_train[0])
        if data_category == 'test':
            label = self.input_queue_test[1]
            image_contents = tf.read_file(self.input_queue_test[0])    
            
        image = tf.image.decode_jpeg(image_contents, channels=self.image_depth)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, [self.image_width, self.image_height])
#        image = tf.image.random_flip_left_right(image)
#        image = tf.image.random_brightness(image, max_delta=0.5)
#        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        
        
        
#        image = tf.image.per_image_standardization(image)   # comment is when test get file and plot

        if self.is_shuffle:
            image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                              batch_size=self.batch_size,
                                                              num_threads=8,
                                                              capacity=2000,
                                                              min_after_dequeue=200)
        else:
            image_batch, label_batch = tf.train.batch([image, label],
                                                      batch_size=self.batch_size,
                                                      num_threads=8,
                                                      capacity=2000)
            
        
        label_batch = tf.one_hot(label_batch, depth=self.n_classes)
        label_batch = tf.cast(label_batch, dtype=tf.int32)
        label_batch = tf.reshape(label_batch, [self.batch_size, self.n_classes])
        return image_batch, label_batch
