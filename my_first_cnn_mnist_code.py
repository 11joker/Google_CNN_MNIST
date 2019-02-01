# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 09:26:44 2019

@author: 25493
"""
#mnist.train.images.shape：(55000, 784)
#mnist.test.images.shape：(10000, 784)
#mnist.validation.images.shape：(5000, 784)


#使用GradientDescentOptimizer
#learing_rate:1e-3

#  放在循环里,速度很慢
#1200 steps accuracy 0.9532
#1500 steps accuracy 0.957
#1600 steps accuracy 0.9588
#  放在循环外，速度快
#9900 steps accuracy 0.9748
#使用0.01,循环到几百次后，0.0958就没有改变，掉坑里了
BATCH_SIZE = 100
IMAGE_SIZE = 28
LABEL_NODE = 10

CHANNEL = 1

CONV1_SIZE = 4
CONV1_DEEP = 2

CONV2_SIZE = 3
CONV2_DEEP = 1

FNN1 = 400

OUTPUT = 10

STEPS = 10000
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
#------------------------------------第二层卷积-最大池--------------------------------------------------------------
    conv2_weights = tf.get_variable("conv2_weights", 
                [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv2 = tf.nn.conv2d(max_pool1, conv2_weights, strides=[1,1,1,1], padding="SAME")
    max_pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name="max_pool2")
#------------------------------------第一层前馈神经网络--------------------
    shape = max_pool2.get_shape().as_list()
    print(shape)
    print(type(max_pool2))
    nodes = shape[1]*shape[2]*shape[3]
    reshaped = tf.reshape(tensor=max_pool2, shape=[-1, nodes])
 
    fnn1_weights = tf.get_variable("fnn1_weights",
                [nodes,FNN1], 
                initializer=tf.truncated_normal_initializer(stddev=0.1))
    fnn1 = tf.matmul(reshaped, fnn1_weights, name="fnn1_op")
    
    bias1 = tf.get_variable("bias1",
            [FNN1],
            initializer=tf.constant_initializer(0.1))
    relu1 = tf.nn.relu(fnn1 + bias1,name="relu1")
#---------------------------------------第二层前馈神经网络------------------------
    fnn2_weights = tf.get_variable("fnn2_weights",
            [FNN1,OUTPUT],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
    fnn2 = tf.matmul(relu1, fnn2_weights, name="fnn2_op")
    bias2 = tf.get_variable("bias2",
            [OUTPUT],
            initializer=tf.constant_initializer(0.1))
    relu2 = tf.nn.relu(fnn2+bias2,name="relu2")
#--------------------------------------------
    return relu2

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
    train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
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




"""
import tensorflow as tf
def main():
    x = tf.Variable(tf.truncated_normal(shape = [100,10],stddev=0.1))
    w = tf.Variable(tf.truncated_normal(shape = [10,5],stddev=0.1))
    bias = tf.Variable(tf.constant(0.1,shape=[5]))
    
    result = tf.matmul(x,w)+bias
    return result

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    result = main()
    print(sess.run(result))
    """
    
    