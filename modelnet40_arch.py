"""
This file is supposed to contain the architecture in each function

If there are new architectures to try a new function should be started for each

e.g. if you're going to try with more capsule layers or conv layers

The motivation is to keep kinds of changes separate from one another

e.g. once you've settled on an architecture, it should be simple to gridsearch/hillclimb to the optimal hyperparamters like learning rate or number of capsules
"""
import numpy as np
import os
import sys

from keras.callbacks import (TensorBoard,
                             EarlyStopping,
                             ReduceLROnPlateau,
                             CSVLogger)
from keras.layers import Conv3D, Dense, Reshape, Add, Input
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.utils import to_categorical, multi_gpu_model
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf

from capsulenet import margin_loss
from capsulelayers import (CapsuleLayer, PrimaryCap,
                           Length, Mask, Conv3DCap)

from data import load_data
from utils import upsample_classes, stratified_shuffle
from results import process_results


NAME= 'ModelNet40'
NUM_EPOCHS = 10

# Load the data
(x_train, y_train), (x_test, y_test), target_names = load_data(NAME)
x_train, y_train, x_val, y_val = stratified_shuffle(x_train, y_train, test_size=.1)
x_train, y_train = upsample_classes(x_train, y_train)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)
if 'test' in sys.argv:
    print('RUNNING IN TEST MODE')
    x_train, y_train, x_val, y_val, x_test, y_test = \
        x_train[0:8192:8], y_train[0:8192:8], x_val[0:2000:20], y_val[0:2000:20], x_test[0:1500:15], y_test[0:1500:15]
    NUM_EPOCHS = 2

n_class = y_test.shape[1]
input_shape = (30, 30, 30, 1)


def base_model(model_name='base_model',
               dim_sub_capsule=16,
               dim_primary_capsule=5,
               n_channels=8,
               primary_cap_kernel_size=9,
               first_layer_kernel_size=9,
               conv_layer_filters=48,
               gpus=1):
    model_name = NAME + '_' + model_name

    def make_model():
        x = Input(shape=input_shape)
        conv1 = Conv3D(filters=conv_layer_filters, kernel_size=first_layer_kernel_size, strides=1,
                                      padding='valid', activation='relu', name='conv1')(x)
        primarycaps = PrimaryCap(conv1, dim_capsule=dim_primary_capsule, n_channels=n_channels,
                                                          kernel_size=primary_cap_kernel_size, strides=2, padding='valid',
                                                          name='primarycap_conv3d')
        sub_caps = CapsuleLayer(num_capsule=n_class, dim_capsule=dim_sub_capsule,
                                                         routings=3, name='sub_caps')(primarycaps)
        out_caps = Length(name='capsnet')(sub_caps)

        # Decoder network
        y = Input(shape=(n_class,))
        masked_by_y = Mask()([sub_caps, y])
        masked = Mask()(sub_caps)

        # shared decoder model in training and prediction
        decoder = Sequential(name='decoder')
        decoder.add(Dense(512, activation='relu',
                          input_dim=dim_sub_capsule*n_class))
        decoder.add(Dense(1024, activation='relu'))
        decoder.add(Dense(np.prod(input_shape), activation='sigmoid'))
        decoder.add(Reshape(target_shape=input_shape, name='out_recon'))


        ### Models for training and evaluation (prediction and actually using)
        train_model = Model([x, y], [out_caps, decoder(masked_by_y)])
        eval_model = Model(x, [out_caps, decoder(masked)])

        ### manipulate model can be used to visualize activation maps for specific classes
        noise = Input(shape=(n_class, dim_sub_capsule))
        noised_sub_caps = Add()([sub_caps, noise])
        masked_noised_y = Mask()([noised_sub_caps, y])
        manipulate_model = Model([x, y, noise], decoder(masked_noised_y))
        return train_model, eval_model, manipulate_model

    ##### If using multiple GPUS ##########
    if gpus > 1:
        with tf.device("/cpu:0"):
            train_model, eval_model, manipulate_model = make_model()
    else:
        train_model, eval_model, manipulate_model = make_model()

    ################################ Compile and Train ###############################
    ##### IF USING MULTIPLE GPUS APPLY JUST BEFORE COMPILING ######
    if gpus > 1:
        train_model = multi_gpu_model(train_model, gpus=gpus) #### Adjust for number of gpus
    # train_model = multi_gpu_model(train_model, gpus=2) #### Adjust for number of gpus
    ##### IF USING MULTIPLE GPUS ######


    INIT_LR = 0.008
    lam_recon = .04
    optimizer = Adam(lr=INIT_LR)
    train_model.compile(optimizer,
                        loss=[margin_loss, 'mse'],
                        loss_weights=[1., lam_recon],
                        metrics={'capsnet': 'accuracy'})

    call_back_path = 'logs/{}.log'.format(model_name)
    tb = TensorBoard(log_dir=call_back_path)
    csv = CSVLogger(os.path.join(call_back_path, 'training.log'))
    early_stop = EarlyStopping(monitor='val_capsnet_acc',
                               min_delta=0,
                               patience=12,
                               verbose=1,
                               mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_capsnet_acc', factor=0.5,
                                  patience=3, min_lr=0.0001)
    train_model.fit([x_train, y_train], [y_train, x_train],
                                    batch_size=32, epochs=NUM_EPOCHS,
                                    validation_data=[[x_val, y_val], [y_val, x_val]],
                    #                 callbacks=[tb, checkpointer])
                                    callbacks=[tb, csv, reduce_lr, early_stop])


    ################################ Process the results ###############################
    process_results(model_name, train_model, eval_model,
                    manipulate_model, x_test, y_test, target_names,
                    INIT_LR=INIT_LR,
                    lam_recon=lam_recon,
                    NUM_EPOCHS=NUM_EPOCHS,
                    dim_sub_capsule=dim_sub_capsule,
                    dim_primary_capsule=dim_primary_capsule,
                    n_channels=n_channels,
                    primary_cap_kernel_size=primary_cap_kernel_size,
                    first_layer_kernel_size=first_layer_kernel_size,
                    conv_layer_filters=conv_layer_filters)


def two_convcaps_layers(model_name='two_convcaps_layers',
                        conv_layer_filters=48,
                        first_layer_kernel_size=9,
                        primary_cap_kernel_size=5,
                        dim_primary_capsule=5,
                        n_channels=5,
                        primary_2_cap_kernel_size=5,
                        dim_primary_capsule_2=5,
                        n_channels_2=5,
                        dim_sub_capsule=16,
                        gpus=1):
    model_name = NAME + '_' + model_name

    def make_model():
        x = Input(shape=input_shape)
        conv1 = Conv3D(filters=conv_layer_filters, kernel_size=first_layer_kernel_size, strides=1,
                                      padding='valid', activation='relu', name='conv1')(x)
        primarycaps = Conv3DCap(conv1, dim_capsule=dim_primary_capsule, n_channels=n_channels,
                                kernel_size=primary_cap_kernel_size, strides=2, padding='valid',
                                name='primarycap_conv3d')
        secondary_caps = PrimaryCap(primarycaps, dim_capsule=dim_primary_capsule_2, n_channels=n_channels_2,
                                    kernel_size=primary_2_cap_kernel_size, strides=1, padding='valid',
                                    name='secondarycap_conv3d')
        sub_caps = CapsuleLayer(num_capsule=n_class, dim_capsule=dim_sub_capsule,
                                                         routings=3, name='sub_caps')(secondary_caps)

        out_caps = Length(name='capsnet')(sub_caps)

        # Decoder network
        y = Input(shape=(n_class,))
        masked_by_y = Mask()([sub_caps, y])
        masked = Mask()(sub_caps)

        # shared decoder model in training and prediction
        decoder = Sequential(name='decoder')
        decoder.add(Dense(512, activation='relu',
                          input_dim=dim_sub_capsule*n_class))
        decoder.add(Dense(1024, activation='relu'))
        decoder.add(Dense(np.prod(input_shape), activation='sigmoid'))
        decoder.add(Reshape(target_shape=input_shape, name='out_recon'))


        ### Models for training and evaluation (prediction and actually using)
        train_model = Model([x, y], [out_caps, decoder(masked_by_y)])
        eval_model = Model(x, [out_caps, decoder(masked)])

        ### manipulate model can be used to visualize activation maps for specific classes
        noise = Input(shape=(n_class, dim_sub_capsule))
        noised_sub_caps = Add()([sub_caps, noise])
        masked_noised_y = Mask()([noised_sub_caps, y])
        manipulate_model = Model([x, y, noise], decoder(masked_noised_y))
        return train_model, eval_model, manipulate_model

    ##### If using multiple GPUS ##########
    if gpus > 1:
        with tf.device("/cpu:0"):
            train_model, eval_model, manipulate_model = make_model()
    else:
        train_model, eval_model, manipulate_model = make_model()

    ################################ Compile and Train ###############################
    ##### IF USING MULTIPLE GPUS APPLY JUST BEFORE COMPILING ######
    if gpus > 1:
        train_model = multi_gpu_model(train_model, gpus=gpus) #### Adjust for number of gpus
    # train_model = multi_gpu_model(train_model, gpus=2) #### Adjust for number of gpus
    ##### IF USING MULTIPLE GPUS ######


    INIT_LR = 0.005
    lam_recon = .04
    optimizer = Adam(lr=INIT_LR)
    train_model.compile(optimizer,
                        loss=[margin_loss, 'mse'],
                        loss_weights=[1., lam_recon],
                        metrics={'capsnet': 'accuracy'})

    call_back_path = 'logs/{}.log'.format(model_name)
    tb = TensorBoard(log_dir=call_back_path)
    csv = CSVLogger(os.path.join(call_back_path, 'training.log'))
    early_stop = EarlyStopping(monitor='val_capsnet_acc',
                               min_delta=0,
                               patience=12,
                               verbose=1,
                               mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_capsnet_acc', factor=0.5,
                                  patience=3, min_lr=0.0001)
    train_model.fit([x_train, y_train], [y_train, x_train],
                                    batch_size=512, epochs=NUM_EPOCHS,
                                    validation_data=[[x_val, y_val], [y_val, x_val]],
                    #                 callbacks=[tb, checkpointer])
                                    callbacks=[tb, reduce_lr, early_stop])


    ################################ Process the results ###############################
    process_results(model_name, eval_model,
                    manipulate_model, x_test, y_test, target_names,
                    INIT_LR=INIT_LR,
                    lam_recon=lam_recon,
                    NUM_EPOCHS=NUM_EPOCHS,
                    dim_sub_capsule=dim_sub_capsule,
                    dim_primary_capsule=dim_primary_capsule,
                    n_channels=n_channels,
                    primary_cap_kernel_size=primary_cap_kernel_size,
                    first_layer_kernel_size=first_layer_kernel_size,
                    conv_layer_filters=conv_layer_filters)


def main():
    from sklearn.model_selection import ParameterGrid
    param_grid = {
        "first_layer_kernel_size": [9],
        "conv_layer_filters": [24, 48],
        "primary_cap_kernel_size": [9, 5],
        "dim_primary_capsule": [4, 8],
        "n_channels": [4, 8],
        "dim_sub_capsule": [8, 16],
    }
    param_grid = ParameterGrid(param_grid)

    for params in param_grid:
        try:
            base_model(model_name='base_model',
                       gpus=8,
                       **params)
        except:
            print('whoops')

    # base_model(model_name='base_model',
    #            first_layer_kernel_size=9,
    #            conv_layer_filters=48,
    #            primary_cap_kernel_size=9,
    #            dim_primary_capsule=5,
    #            n_channels=8,
    #            dim_sub_capsule=16,
    #            gpus=8)

    # two_convcaps_layers(model_name='two_convcaps_layers',
    #                     conv_layer_filters=48,
    #                     first_layer_kernel_size=3,
    #                     primary_cap_kernel_size=9,
    #                     dim_primary_capsule=4,
    #                     n_channels=6,
    #                     primary_2_cap_kernel_size=5,
    #                     dim_primary_capsule_2=4,
    #                     n_channels_2=5,
    #                     dim_sub_capsule=16,
    #                     gpus=8)

if __name__ == '__main__':
    main()

