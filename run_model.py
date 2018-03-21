import numpy as np
import scipy as sp

import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Add, Conv3D, Dense, Input, Reshape
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import multi_gpu_model, to_categorical

from data import load_data
from capsulelayers import CapsuleLayer, Length, Mask, PrimaryCap
from capsulenet import margin_loss
from utils import stratified_shuffle, upsample_classes

# Load the data
(x_train, y_train), (x_test, y_test), target_names = load_data('./ModelNet40/')
x_train, y_train, x_val, y_val = stratified_shuffle(x_train, y_train, test_size=.1)
x_train, y_train = upsample_classes(x_train, y_train)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

n_class = y_test.shape[1]
input_shape = (30, 30, 30, 1)

dim_sub_capsule = 16
dim_primary_capsule = 8
n_channels = 4
primary_cap_kernel_size = 9

first_layer_kernel_size = 9
conv_layer_filters = 48

#### Build us a model
with tf.device("/cpu:0"):
    x = Input(shape=(30, 30, 30, 1))

    conv1 = Conv3D(filters=conv_layer_filters, kernel_size=first_layer_kernel_size,
                   strides=1, padding='valid', activation='relu', name='conv1')(x)

    primarycaps = PrimaryCap(conv1, dim_capsule=dim_primary_capsule, n_channels=n_channels,
                             kernel_size=primary_cap_kernel_size, strides=2, padding='valid')

    sub_caps = CapsuleLayer(num_capsule=n_class, dim_capsule=dim_sub_capsule,
                             routings=3, name='sub_caps')(primarycaps)

    out_caps = Length(name='capsnet')(sub_caps)

    y = Input(shape=(n_class,))
    masked_by_y = Mask()([sub_caps, y])

    masked = Mask()(sub_caps)

    decoder = Sequential(name='decoder')
    decoder.add(Dense(512, activation='relu',
                      input_dim=dim_sub_capsule*n_class))
    # decoder.add(Dense(64, activation='relu'))
    decoder.add(Dense(1024, activation='relu'))
    decoder.add(Dense(np.prod(input_shape), activation='sigmoid'))

    decoder.add(Reshape(target_shape=input_shape, name='out_recon'))

    train_model = Model([x, y], [out_caps, decoder(masked_by_y)])


    eval_model = Model(x, [out_caps, decoder(masked)])
    similarity_model = Model(x, [])

    ### build a manipulate model
    noise = Input(shape=(n_class, dim_sub_capsule))
    noised_sub_caps = Add()([sub_caps, noise])
    masked_noised_y = Mask()([noised_sub_caps, y])
    manipulate_model = Model([x, y, noise], decoder(masked_noised_y))

# compile and train the model
NUM_EPOCHS = 300
INIT_LR = 0.0001
lam_recon = .04
optimizer = Adam(lr=INIT_LR)

train_model = multi_gpu_model(train_model, gpus=2)
train_model.compile(optimizer,
                    loss=[margin_loss, 'mse'],
                    loss_weights=[1., lam_recon],
                    metrics={'capsnet': 'accuracy'})

checkpointer = ModelCheckpoint(filepath='example.hdf5',
                               verbose=1, save_best_only=True)
tb = TensorBoard(log_dir='logs/capsnet_modelnet40.log/')

train_model.fit([x_train, y_train], [y_train, x_train],
                                batch_size=256, epochs=NUM_EPOCHS,
                                validation_data=[[x_val, y_val], [y_val, x_val]],
                #                 callbacks=[tb, checkpointer])
                                callbacks=[tb])


y_pred, x_recon = eval_model.predict(x_test)

print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

train_model.save_weights('modelnet40.last.hdf5')
