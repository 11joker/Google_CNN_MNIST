# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 12:13:00 2019

@author: 25493
"""
import random
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt 

def main(_):
    mnist = input_data.read_data_sets("./")
    train_length = len(mnist.train.images)
    for i in range(10):
        i = random.randint(0, train_length)
        image = mnist.train.images[i]
        label = mnist.train.labels[i]
        im = image.reshape(28, 28)
        plt.title(label)
        plt.imshow(im, cmap="gray")
        plt.show()
        
if __name__=="__main__":
    tf.app.run()

mnist = input_data.read_data_sets("./")
train_length = len(mnist.train.images)
for i in range(10):
    i = random.randint(0, train_length)
    image = mnist.train.images[i]
    label = mnist.train.labels[i]
    im = image.reshape(28, 28)
    print(label)
    print(image)