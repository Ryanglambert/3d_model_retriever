import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import binvox_rw_py as binvox


def plot_class_balance():
    "Visually inspect the balancing strategy is working"
    import matplotlib.pyplot as plt
    import pandas as pd

    from data import load_data
    from utils import upsample_classes, stratified_shuffle

    (x_train, y_train), (x_test, y_test), target_names = load_data('./ModelNet10/')
    x_train, y_train, x_val, y_val = stratified_shuffle(x_train, y_train, test_size=.2)

    train = pd.DataFrame(y_train)
    valid = pd.DataFrame(y_val)
    # inspect that the validation set is representative of the training set
    train[0].value_counts().sort_values().plot(kind='bar', title='Train Labels')
    plt.figure()
    valid[0].value_counts().sort_values().plot(kind='bar', title='Validation Labels')

    # inspect that the Training classes are balanced
    x_train, y_train = upsample_classes(x_train, y_train)
    train = pd.DataFrame(y_train)
    plt.figure()
    train[0].value_counts().sort_values().plot(kind='bar', title='Training Class Balance After Upsampling')


def plot_vox(mat, title='Title'):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(45, 135)
    ax.voxels(mat, edgecolor='k')
    plt.title(title)
    plt.show()


def plot_vox_file(path):
    with open(path, 'rb') as f:
        model = binvox.read_as_3d_array(f)
    # model.data is a 3D boolean array in this case. Probably could also be float array.
    plot_vox(model.data)



def plot_learning_curves(history, epochs=200, y_min=0):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.legend()
    plt.xlim(0, epochs)
    plt.ylim(y_min, 1)
    plt.subplot(212)
    plt.plot(history.history['val_acc'], label='Val Accuracy')
    plt.plot(history.history['acc'], label = 'Training Accuracy')
    plt.legend()
    plt.xlim(0, epochs)
    plt.ylim(y_min, 1)


def _prediction(x, y, target_names, model):
    truth = target_names[np.argmax(y)]
    proba = model.predict(x)
    proba_idx = np.argmax(proba)
    predicted_name = target_names[proba_idx]
    output = ("Model predicts a: {1} with {2:.2f} confidence\n"
              "This is a {0}").format(
                  truth, predicted_name, proba[0][proba_idx])
    return output


def plot_rotation_issue(x, y, target_names, model=None, angle=0, axes=(0, 1)):
    "plot performance before and after with rotation"
    # do 90 degree rotations first
    n_90_rotations = angle // 90
    degrees_from_90 = np.mod(angle, 90)
    x_rotated = np.rot90(x.reshape(30, 30, 30), n_90_rotations, axes=axes)
    # do sub 90 degree rotations (weird behavior using interpolation near mod 90 == 0)
    # something weird happens when I use interpolation close to 90
    if degrees_from_90 > 0 and degrees_from_90 < 90:
        x_rotated = sp.ndimage.interpolation.rotate(x_rotated, degrees_from_90, axes, reshape=False)
    
    # plot the normal model along with its prediction
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.view_init(45, 135)
    ax.voxels(x.reshape(30, 30, 30), edgecolor='k')
    plt.title(_prediction(x, y, target_names, model)) if model else plt.title("normal")

    # plot the rotated model along with its prediction
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.view_init(45, 135)
    ax2.voxels(x_rotated, edgecolor='k')
    rotated_title = "rotated by {} on {} axes".format(
        angle, axes)

    # The title of the rotated includes the prediction if a model was passed
    if model:
        full_title = rotated_title + "\n" + \
            _prediction(x_rotated.reshape(1, 30, 30, 30, 1), y, target_names, model)
    else:
        full_title = rotated_title
    plt.title(full_title)
    plt.show()


def _prediction_capsnet(x, y, target_names, model):
    truth = target_names[np.argmax(y)]
    proba, x_recon = model.predict(x)
    proba_idx = np.argmax(proba)
    predicted_name = target_names[proba_idx]
    output = ("Model predicts a: {1} with {2:.2f} confidence\n"
              "This is a {0}").format(
                  truth, predicted_name, proba[0][proba_idx])
    return output


def plot_capsnet_rotation_issue(x, y, target_names, model=None, angle=0, axes=(0, 1)):
    "plot performance before and after with rotation"
    # do 90 degree rotations first
    n_90_rotations = angle // 90
    degrees_from_90 = np.mod(angle, 90)
    x_rotated = np.rot90(x.reshape(30, 30, 30), n_90_rotations, axes=axes)
    # do sub 90 degree rotations (weird behavior using interpolation near mod 90 == 0)
    # something weird happens when I use interpolation close to 90
    if degrees_from_90 > 0 and degrees_from_90 < 90:
        x_rotated = sp.ndimage.interpolation.rotate(x_rotated, degrees_from_90, axes, reshape=False)
    
    # plot the normal model along with its prediction
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.view_init(45, 135)
    ax.voxels(x.reshape(30, 30, 30), edgecolor='k')
    plt.title(_prediction_capsnet(x, y, target_names, model)) if model else plt.title("normal")

    # plot the rotated model along with its prediction
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.view_init(45, 135)
    ax2.voxels(x_rotated, edgecolor='k')
    rotated_title = "rotated by {} on {} axes".format(
        angle, axes)

    # The title of the rotated includes the prediction if a model was passed
    if model:
        full_title = rotated_title + "\n" + \
            _prediction_capsnet(x_rotated.reshape(1, 30, 30, 30, 1), y, target_names, model)
    else:
        full_title = rotated_title
    plt.title(full_title)
    plt.show()

def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)


def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded


def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z


def plot_shaded(arr_shaded, angle=320, exploded=True, lims=(0, 60)):
    facecolors = cm.gist_yarg(arr_shaded)
    facecolors[:,:,:,-1] = arr_shaded
    facecolors = explode(facecolors)

    filled = facecolors[:,:,:,-1] != 0
    if exploded:
        x, y, z = np.indices(np.array(filled.shape) + 1)
    else:
        x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_zlim(lims)

    ax.voxels(x, y, z, filled, facecolors=facecolors)
    plt.show()
