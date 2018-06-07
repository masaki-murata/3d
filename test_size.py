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


# define autoencoder
def autoencoder(input_shape=(512,256,256,1)):
    input_img = Input(shape=input_shape)
    x = Conv3D(filters=8, kernel_size=(2,2,2), strides=(2,2,2), padding="same", activation="relu")(input_img)
    
    # decoding
    x = UpSampling3D(size=(2,2,2))(x)
    decoded = Conv3D(filters=1, kernel_size=(2,2,2), padding="same", activation="relu")(x)
    
    model = Model(input_img, decoded)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()
    
    return model

# mnist 画像を３次元に埋め込む
def embed_mnist(mnist_image=np.array([]),
                data_shape=(512,256,256),
                slices=28,
                ):
    z,y,x = data_shape[0]//2,data_shape[1]//2,data_shape[2]//2
    embeded = np.zeros(data_shape)
    for z_pos in range(slices):
        embeded[z-slices//2+z_pos, y-14:y+14, x-14:x+14] = mnist_image
    
    return embeded

#def train(data_shape=(512,256,256)):
#    batch_size = 32
#    (X_train, y_train), (X_test, y_test) = mnist.load_data()
#    x_train = np.zeros((batch_size,)+data_shape)
#    x_test = np.zeros((batch_size,)+data_shape)
#    for train_id in range(batch_size):
#        x_train[train_id] = embed_mnist(mnist_image=X_train[train_id],
#                                        data_shape=(512,256,256),
#                                        slices=28)
#    for test_id in range(batch_size):
#        x_test[test_id] = embed_mnist(mnist_image=X_test[test_id],
#                                        data_shape=(512,256,256),
#                                        slices=28)
#    print(np.sum(x_train))
#    print(x_train.shape)

# validation data を作成する関数
def make_validation_data(mnist_x_test,
                         mnist_y_test,
                         data_shape=(512,256,256),
                         slices=28,
                         val_size=100,                          
                         ):
    val_data = np.zeros((val_size,)+data_shape)
    val_label = np.zeros((val_size,))
    for mnist_id in range(val_size):
        val_data[mnist_id] = embed_mnist(mnist_image=mnist_x_test[mnist_id],
                data_shape=data_shape,
                slices=slices,
                )
        val_label[mnist_id] = mnist_y_test[mnist_id]
    val_data = val_data.reshape(val_data.shape+(1,))
    val_label = val_label.reshape(val_label.shape+(1,))
            
    return val_data, val_label

# generator 
def batch_iter(mnist_x_train=np.array([]), 
               mnist_y_train=np.array([]), 
               slices=28,
               data_shape=(512,256,256),
               steps_per_epoch=32,
               batch_size=1,
               ):
        
    while True:
        for step in range(steps_per_epoch):
            data = np.zeros( (batch_size,)+data_shape, dtype=np.uint8 )
            labels = np.zeros( (batch_size,)+data_shape, dtype=np.uint8 )
            for count in range(batch_size):
                data[count] = embed_mnist(mnist_image=mnist_x_train[count],
                                          data_shape=data_shape,
                                          slices=slices,
                                          )
                labels[count] = mnist_y_train[count]
            data = data.reshape(data.shape+(1,))
            labels = labels.reshape(labels.shape+(1,))
#            print("data.shape = ", data.shape)
            yield data, labels
    
    
    
# オートエンコーダをトレーニングする関数
def train_autoencoder(batch_size=32,
                      nb_gpus=4,
                      data_shape=(512,256,256),
                      slices=28,
                      steps_per_epoch=32,
                      epochs=64,
                      val_size=100,
                      ):
    # setting model
    input_shape = data_shape + (1,)
    model_single_gpu = autoencoder(input_shape=input_shape)
    if nb_gpus>1:
        model = multi_gpu_model(model_single_gpu, gpus=nb_gpus)
    elif nb_gpus==1:
        model = model_single_gpu

    # load mnist dataset
    mnist_x_train = np.load("./mnist_x_train.npy")
    mnist_y_train = np.load("./mnist_y_train.npy")
    mnist_x_test = np.load("./mnist_x_test.npy")
    mnist_y_test = np.load("./mnist_y_test.npy")
#    (mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()
    
    # set validation data
    val_data, val_label = make_validation_data(mnist_x_test=mnist_x_test,
                                              mnist_y_test=mnist_y_test,
                                              data_shape=data_shape,
                                              slices=slices,
                                              val_size=val_size,
                                              )
    print("val_data.shape = ", val_data.shape)
    
    # set generator for training data
    train_gen = batch_iter(mnist_x_train=mnist_x_train,
                           mnist_y_train=mnist_y_train, 
                           slices=slices,
                           data_shape=data_shape,
                           steps_per_epoch=steps_per_epoch,
                           batch_size=batch_size,
                           )
    
    # for ループで 1 epoch ずつ学習
    for epoch in range(1,epochs+1):
        model.fit_generator(train_gen,
                            steps_per_epoch=steps_per_epoch,
                            epochs=1,
#                            validation_data=(val_data,val_label)
                            )
        print('Epoch %s/%s done' % (epoch, epochs))
        print("")
        
        # save model
#        if epoch>0 and epoch % 1==0:
#            print(epoch)
#            model_single_gpu.save(path_to_save_model % (epoch))
#            model_single_gpu.save_weights(path_to_save_weights % (epoch))
    
    
    

def main():
#    print("aho")
    train_autoencoder(batch_size=1,
                      nb_gpus=1,
                      data_shape=(512,256,256),
                      slices=28,
                      steps_per_epoch=32,
                      epochs=64,
                      val_size=100,
                      )
    
if __name__ == '__main__':
    main()
