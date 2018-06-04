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

def embed_mnist(mnist_image=np.array([]),
                data_shape=(512,256,256),
                slices=28,
                ):
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

# validation data を作成する関数
def make_validation_data(data_shape=(512,256,256),
                         slices=28,
                         val_size=100,
                         mnist_x_test, 
                         mnist_y_test,
                         ):
    val_data = np.zeros((val_size,)+data_shape)
    for mnist_id in range(val_size):
        val_data[mnist_id] = embed_mnist(mnist_image=mnist_x_test[mnist_id],
                data_shape=data_shape,
                slices=slices,
                )
            
    return val_data, val_label

# オートエンコーダをトレーニングする関数
def train_autoencoder(batch_size=32,
                      nb_gpus=4,
                      data_shape=(512,256,256)):
    input_shape = data_shape + (1,)
    model_single_gpu = autoencoder(input_shape=input_shape)
    model = multi_gpu_model(model_single_gpu, gpus=nb_gpus)

    (mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()
    
    # validation data を作る
    val_data,val_label = make_validation_data(data_shape=(512,256,256),
                         slices=28,
                         mnist_x_test, 
                         mnist_y_test
                         )
    
    # training data の generator をセット
    train_gen = batch_iter(images=train_images,
                           segmentation_gts=train_gts, 
                           crop_shape=crop_shape,
                           steps_per_epoch=steps_per_epoch,
                           batch_size=batch_size,
                           )
    
    # for ループで 1 epoch ずつ学習
    for epoch in range(1,epochs+1):
        model.fit_generator(train_gen,
                            steps_per_epoch=steps_per_epoch,
                            epochs=1,
                            validation_data=(val_data,val_label)
                            )
        print('Epoch %s/%s done' % (epoch, epochs))
        print("")
        
        if epoch>0 and epoch % 1==0:
            print(epoch)
            model_single_gpu.save(path_to_save_model % (epoch))
            model_single_gpu.save_weights(path_to_save_weights % (epoch))
    
    
    

def main():
    print("aho")
    train()
    
if __name__ == '__main__':
    main()
