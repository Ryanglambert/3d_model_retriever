import numpy as np
import scipy as sp
# from keras import layers, models, optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Conv3D, Dense, Reshape, Add, Input
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.utils import to_categorical, multi_gpu_model
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

from capsulenet import margin_loss
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

from data import load_data
from plots import (plot_vox,
                                      plot_capsnet_rotation_issue,
                                      plot_rotation_issue,
                                      plot_shaded,
                                      plot_dots,
                                      plot_recons)
from utils import upsample_classes, stratified_shuffle


(x_train, y_train), (x_test, y_test), target_names = load_data('./ModelNet10/')
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


##### If using multiple GPUS ##########
# with tf.device("/cpu:0"):
x = Input(shape=(30, 30, 30, 1))
conv1 = Conv3D(filters=conv_layer_filters, kernel_size=first_layer_kernel_size, strides=1,
                              padding='valid', activation='relu', name='conv1')(x)
primarycaps = PrimaryCap(conv1, dim_capsule=dim_primary_capsule, n_channels=n_channels,
                                                  kernel_size=primary_cap_kernel_size, strides=2, padding='valid',
                                                  name='primarycap_conv3d')
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


### Different kinds of models
train_model = Model([x, y], [out_caps, decoder(masked_by_y)])
eval_model = Model(x, [out_caps, decoder(masked)])

### manipulate model
noise = Input(shape=(n_class, dim_sub_capsule))
noised_sub_caps = Add()([sub_caps, noise])
masked_noised_y = Mask()([noised_sub_caps, y])
manipulate_model = Model([x, y, noise], decoder(masked_noised_y))



### Training
NUM_EPOCHS = 22
INIT_LR = .0001

optimizer = Adam(lr=INIT_LR)
lam_recon = .04
##### IF USING MULTIPLE GPUS ######
# train_model = multi_gpu_model(train_model, gpus=8) #### Adjust for number of gpus
# train_model = multi_gpu_model(train_model, gpus=2) #### Adjust for number of gpus
##### IF USING MULTIPLE GPUS ######
train_model.compile(optimizer,
                    loss=[margin_loss, 'mse'],
                    loss_weights=[1., lam_recon],
                    metrics={'capsnet': 'accuracy'})

tb = TensorBoard(log_dir='logs/capsnet_modelnet10.log/')
train_model.fit([x_train, y_train], [y_train, x_train],
                                batch_size=256, epochs=NUM_EPOCHS,
                                validation_data=[[x_val, y_val], [y_val, x_val]],
                #                 callbacks=[tb, checkpointer])
                                callbacks=[tb])

### Get test accuracy
y_pred, x_recon = eval_model.predict(x_test)
test_accuracy = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
print('Test acc:', test_accuracy)

print('Saving Models...')
train_model.save('models/train_model_net10_acc_{}.hdf5'.format(str(round(test_accuracy, 5)).replace('.', '')))
eval_model.save('models/eval_model_net10_acc_{}.hdf5'.format(str(round(test_accuracy, 5)).replace('.', '')))
manipulate_model.save('models/manipulate_model_net10_acc_{}.hdf5'.format(str(round(test_accuracy, 5)).replace('.', '')))

