# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 12:04:03 2019

@author: 25493
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 09:26:44 2019

@author: 25493
"""
#mnist.train.images.shape：(55000, 784)
#mnist.test.images.shape：(10000, 784)
#mnist.validation.images.shape：(5000, 784)


#使用GradientDescentOptimizer CONV1_SIZE = 4 CONV1_DEEP = 2
#learing_rate:1e-3
#0.92
#60000 steps CONV1_DEEP:30 CONV1_SIZE:5 accuracy:0.95
BATCH_SIZE = 100
IMAGE_SIZE = 28
LABEL_NODE = 10

CHANNEL = 1

CONV1_SIZE = 5
CONV1_DEEP = 30

CONV2_SIZE = 3
CONV2_DEEP = 1

FNN1 = 400

OUTPUT = 10

STEPS = 60000
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def train(x):
#-----------------------------------第一层卷积-最大池---------------------------------------------------------------    
    conv1_weights = tf.get_variable("conv1_weights",
        [CONV1_SIZE, CONV1_SIZE, CHANNEL, CONV1_DEEP],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv1 = tf.nn.conv2d(x, conv1_weights, strides=[1,1,1,1], padding="SAME")
    
    max_pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name="max_pool1")

#------------------------------------第一层前馈神经网络--------------------
    shape = max_pool1.get_shape().as_list()
    print(shape)
    print(type(max_pool1))
    nodes = shape[1]*shape[2]*shape[3]
    reshaped = tf.reshape(tensor=max_pool1, shape=[-1, nodes])
 
    fnn1_weights = tf.get_variable("fnn1_weights",
                [nodes,10], 
                initializer=tf.truncated_normal_initializer(stddev=0.1))
    fnn1 = tf.matmul(reshaped, fnn1_weights, name="fnn1_op")
    
    bias1 = tf.get_variable("bias1",
            [10],
            initializer=tf.constant_initializer(0.1))
    relu1 = tf.nn.relu(fnn1 + bias1,name="relu1")

#--------------------------------------------
    return relu1

def accuracy(true_y, predict_y):
    correct = tf.equal(tf.argmax(true_y,1), tf.argmax(predict_y,1))
    correct_rate = tf.reduce_mean(tf.cast(correct, tf.float32))
    return correct_rate

def main(_):
    mnist = input_data.read_data_sets("./",one_hot=True)
    validation_x, validation_y = mnist.validation.next_batch(BATCH_SIZE)
    validation_x = np.reshape(validation_x,(BATCH_SIZE,
                              IMAGE_SIZE,
                              IMAGE_SIZE,
                              CHANNEL))
    x = tf.placeholder(dtype = tf.float32, shape=[None, 
                    IMAGE_SIZE, 
                    IMAGE_SIZE, CHANNEL], name = "x_input")
    y_ = tf.placeholder(dtype = tf.float32, shape=[None, LABEL_NODE], name = "y_input")
    y = train(x)
#--------------------------loss-----------------------
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = y, labels = tf.arg_max(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
#--------------------------------------------------
    accuracy_op = accuracy(y_, y)
#-------------------------------------------------------
    validation_x = np.reshape(mnist.validation.images,(-1,
                              IMAGE_SIZE,
                              IMAGE_SIZE,
                              CHANNEL))
    #初次使用0.1学习率导致出现NaN值
    train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy_mean)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(STEPS):
            train_x, train_y = mnist.train.next_batch(BATCH_SIZE)
            train_x = np.reshape(train_x,(BATCH_SIZE,
                              IMAGE_SIZE,
                              IMAGE_SIZE,
                              CHANNEL))
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