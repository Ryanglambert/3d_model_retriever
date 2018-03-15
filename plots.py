import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp

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


def plot_rotation_issue(x, y, target_names, model, angle, axes):
    "Plot performance before and after with rotation"
    x_rotated = sp.ndimage.interpolation.rotate(x.reshape(30, 30, 30), angle, axes, reshape=False)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.view_init(45, 135)
    ax.voxels(x.reshape(30, 30, 30), edgecolor='k')
    plt.title(_prediction(x, y, target_names, model))
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.view_init(45, 135)
    ax2.voxels(x_rotated, edgecolor='k')
    rotated_title = "Rotated by {} on {} axes".format(
        angle, axes)
    full_title = rotated_title + "\n" + \
        _prediction(x_rotated.reshape(1, 30, 30, 30, 1), y, target_names, model)
    plt.title(full_title)
    plt.show()

