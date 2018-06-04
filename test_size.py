# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 15:05:39 2018

@author: murata
"""

import numpy as np
import keras
from keras.layers import Input, Dense, Conv3D, UpSampling3D
from keras.models import Model
from keras.datasets import mnist
from keras.utils.training_utils import multi_gpu_model


def autoencoder(input_shape=(512,256,256,1)):
    input_img = Input(shape=input_shape)
    x = Conv3D(filters=8, kernel_size=(3,3,3), strides=(3,3,3), padding="same", activation="relu")(input_img)
    
    # decoding
    x = UpSampling3D(size=(3,3,3))(x)
    decoded = Conv3D(filters=1, kernel_size=(3,3,3), padding="same", activation="relu")(x)
    
    model = Model(input_img, decoded)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()
    
    return model

def embed_mnist(mnist_image,
                data_shape=(512,256,256),
                slices=28,):
    z,y,x = data_shape[0]//2,data_shape[1]//2,data_shape[2]//2
    embeded = np.zeros(data_shape)
    for z_pos in range(slices):
        embeded[z-slices//2+z_pos, y-14:y+14, x-14:x+14] = mnist_image
    
    return embeded

def train(data_shape=(512,256,256)):
    batch_size = 32
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    x_train = np.zeros((batch_size,)+data_shape)
    x_test = np.zeros((batch_size,)+data_shape)
    for train_id in range(batch_size):
        x_train[train_id] = embed_mnist(mnist_image=X_train[train_id],
                                        data_shape=(512,256,256),
                                        slices=28)
    for test_id in range(batch_size):
        x_test[test_id] = embed_mnist(mnist_image=X_test[test_id],
                                        data_shape=(512,256,256),
                                        slices=28)
    print(np.sum(x_train))
    print(x_train.shape)


def train_autoencoder(batch_size=32,
                      nb_gpus=4,
                      data_shape=(512,256,256)):
    input_shape = data_shape + (1,)
    model_single_gpu = autoencoder(input_shape=input_shape)
    model = multi_gpu_model(model_single_gpu, gpus=nb_gpus)
        

def main():
    print("aho")
    train()
    
if __name__ == '__main__':
    main()
