# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 09:26:44 2019

@author: 25493
"""
#mnist.train.images.shape：(55000, 784)
#mnist.test.images.shape：(10000, 784)
#mnist.validation.images.shape：(5000, 784)

#使用GradientDescentOptimizer 7000 steps 0.926
#使用AdamOptimizer accuracy 0.931
BATCH_SIZE = 500
IMAGE_SIZE = 28
LABEL_NODE = 10

CHANNEL = 1

CONV1_SIZE = 4
CONV1_DEEP = 2

CONV2_SIZE = 3
CONV2_DEEP = 1

FNN1 = 400

OUTPUT = 10

STEPS = 30000

LEARNING_RATE_DECAY = 0.99
LEARNING_RATE = 0.06

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def train(x):
#------------------------------------第一层前馈神经网络--------------------
    nodes = x.shape[1]
    fnn1_weights = tf.get_variable("fnn1_weights",
                [nodes,10], 
                initializer=tf.truncated_normal_initializer(stddev=0.1))
    fnn1 = tf.matmul(x, fnn1_weights, name="fnn1_op")
    
    bias1 = tf.get_variable("bias1",
            [10],
            initializer=tf.constant_initializer(0.1))
    relu = tf.nn.relu(fnn1 + bias1,name="relu1")
    return relu

def accuracy(true_y, predict_y):
    correct = tf.equal(tf.argmax(true_y,1), tf.argmax(predict_y,1))
    correct_rate = tf.reduce_mean(tf.cast(correct, tf.float32))
    return correct_rate

def main(_):
    mnist = input_data.read_data_sets("./",one_hot=True)
    validation_x, validation_y = mnist.validation.next_batch(100)

    x = tf.placeholder(dtype = tf.float32, shape=[None, 784], name = "x_input")
    y_ = tf.placeholder(dtype = tf.float32, shape=[None, LABEL_NODE], name = "y_input")
    y = train(x)
#--------------------------loss-----------------------
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = y, labels = tf.arg_max(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
#--------------------------------------------------
    accuracy_op = accuracy(y_, y)
#-------------------------------------------------------
    validation_x = np.reshape(mnist.validation.images,(-1,784))
    #global step
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy_mean)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(STEPS):
            train_x, train_y = mnist.train.next_batch(BATCH_SIZE)
            train_x = np.reshape(train_x,(BATCH_SIZE,784))
#初次使用0.1学习率导致出现NaN值
#之后使用0.001 学习速度增快，很快达到了0.91
#之后使用exponential_decay效果不好
            _, loss = sess.run([train_step, cross_entropy_mean],
                     feed_dict = {x:train_x, y_:train_y})
            if(i%100==0):
                print("After %d step,the training loss is: %g"%(i,loss))
                loss, accuracy_rate = sess.run([cross_entropy_mean, accuracy_op],
                feed_dict={x:validation_x, y_:mnist.validation.labels})
                print("After %d step,the validation loss is: %g"%(i,loss))
                print("ACCURACY  After %d step,the accuracy loss is: %g\n"%(i,accuracy_rate))               

if __name__=="__main__":
    tf.app.run()


