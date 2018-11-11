import tensorflow as tf
import numpy as np
from lib.base_network import Net


class NetWork(Net):
    def __init__(self, cfg_):
        super().__init__(cfg_)
        self.x = tf.placeholder(tf.float32, name='x', shape=[self.config.batch_size,
                                                             self.config.image_width,
                                                             self.config.image_height,
                                                             self.config.image_depth], )
        self.y = tf.placeholder(tf.int16, name='y', shape=[self.config.batch_size,
                                                           self.config.n_classes])
        self.loss = None
        self.accuracy = None
        self.summary = []
        self.drop=self.config.drop_out

    def init_saver(self):
        pass

    def get_summary(self):
        return self.summary

    def conv2d(self, layer_name, inputs, out_channels, kernel_size, strides=1, padding='SAME'):
        in_channels = inputs.get_shape()[-1]
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
            w = tf.get_variable(name='weights',
                                trainable=True,
                                shape=[kernel_size, kernel_size, in_channels, out_channels],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='biases',
                                trainable=True,
                                shape=[out_channels],
                                initializer=tf.constant_initializer(0.0))
            inputs = tf.nn.conv2d(inputs, w, [1, strides, strides, 1], padding=padding, name='conv')
            inputs = tf.nn.bias_add(inputs, b, name='bias_add')
            inputs = tf.nn.relu(inputs, name='relu')
            return inputs

    def max_pool(self, layer_name, inputs, pool_size, strides, padding='SAME'):
        with tf.name_scope(layer_name):
            return tf.nn.max_pool(inputs, [1, pool_size, pool_size, 1], [1, strides, strides, 1], padding=padding,
                                  name=layer_name)


    def dropout(self, layer_name, inputs, keep_prob):
        # dropout_rate = 1 - keep_prob
        with tf.name_scope(layer_name):
            return tf.nn.dropout(name=layer_name, x=inputs, keep_prob=keep_prob)


    def fc(self, layer_name, inputs, out_nodes):
        shape = inputs.get_shape()
        if len(shape) == 4:  # x is 4D tensor
            size = shape[1].value * shape[2].value * shape[3].value
        else:  # x has already flattened
            size = shape[-1].value
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
            w = tf.get_variable('weights',
                                shape=[size, out_nodes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('biases',
                                shape=[out_nodes],
                                initializer=tf.constant_initializer(0.0))
            flat_x = tf.reshape(inputs, [-1, size])
            inputs = tf.nn.bias_add(tf.matmul(flat_x, w), b)
            inputs = tf.nn.relu(inputs)
            return inputs

    def cal_loss(self, logits, labels):
        with tf.name_scope('loss') as scope:
#            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
#                logits=logits, labels=labels, name='cross-entropy')
#            pos_weights=tf.constant([3.0,1.0])
#            cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels,
#                                                                     pos_weight=pos_weights,name='cross_entropy')
#            
#            self.loss = tf.reduce_mean(cross_entropy, name='loss')
            
            class_weights = tf.constant([[3.0,1.0]])
            weights = tf.reduce_sum(class_weights * tf.to_float(labels), axis=1)
            unweighted_losses = tf.nn.softmax_cross_entropy_with_logits( logits=logits, labels=labels)
            weighted_losses = unweighted_losses * weights
            self.loss = tf.reduce_mean(weighted_losses)
#            self.loss = tf.losses.softmax_cross_entropy()
            loss_summary = tf.summary.scalar(scope, self.loss)
            self.summary.append(loss_summary)

    def cal_accuracy(self, logits, labels):
        with tf.name_scope('accuracy') as scope:
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            correct = tf.cast(correct, tf.float32)
            self.accuracy = tf.reduce_mean(correct) * 100.0
            accuracy_summary = tf.summary.scalar(scope, self.accuracy)
            self.summary.append(accuracy_summary)

    def optimize(self):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
#            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            train_op = optimizer.minimize(self.loss, global_step=self.global_step_tensor)
            return train_op

    def build_model(self):

        conv1 = self.conv2d('conv1', self.x, 32, 3,1)
        pool1 = self.max_pool('pool1', conv1, 3, 2)

        conv2 = self.conv2d('conv2', pool1, 64, 3,1)
        pool2 = self.max_pool('pool2', conv2, 3, 2)

        conv3 = self.conv2d('conv3', pool2, 64, 3,1)
        pool3 = self.max_pool('pool3', conv3, 3, 2)
        
        conv4 = self.conv2d('conv4', pool3, 64, 3,1)
        pool4 = self.max_pool('pool4', conv4, 3, 2)
        
 

    #    import pdb; pdb.set_trace()
        fc5 = self.fc('fc5',pool4,out_nodes=128)
        fc5_drop = self.dropout('fc5_drop',fc5,self.drop)
    
        fc6 = self.fc('fc6',fc5_drop,out_nodes=64)
        fc6_drop = self.dropout('fc6_drop',fc6,self.drop)
        
        
        self.logits = self.fc('loss_classifier', fc6_drop, out_nodes=self.config.n_classes)

        self.cal_loss(self.logits, self.y)
        self.cal_accuracy(self.logits, self.y)

