# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:42:57 2019

@author: 25493
"""
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS =10

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

#全连接层
FC_SIZE =512

def inference(input_tensor,train,regularizer):
    with tf.variable_scope("layer1-conv1"):
        conv1_weights = tf.get_variable(
                "weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(
                "bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(
                input_tensor,conv1_weights,strides=[1,1,1,1],padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(
                relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
                "weight",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable(
                "bias",[CONV2_DEEP],
                initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(
                pool1,conv2_weights,strides=[1,1,1,1],padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(
                relu2,ksize = [1,2,2,1], strides = [1,2,2,1],
                padding = "SAME")
    pool_shape = pool2.get_shape().as_list()
    
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])
    
    with tf.variable_scope("layer5-fcl"):
        fcl_weights = tf.get_variable(
                "weight",[nodes, FC_SIZE],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("losses",regularizer(fcl_weights))
        fcl_biases = tf.get_variable(
                "bias",[FC_SIZE],initializer=tf.constant_initializer(0.1))
        fcl = tf.nn.relu(tf.matmul(reshaped, fcl_weights) + fcl_biases)
        if train:fcl = tf.nn.dropout(fcl,0.5)
    
    with tf.variable_scope("layer-fc2"):
        fc2_weights = tf.get_variable(
                "weight",[FC_SIZE,NUM_LABELS],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("losses",regularizer(fc2_weights))
        fc2_biases = tf.get_variable(
                "bias",[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fcl,fc2_weights) + fc2_biases
    return logit