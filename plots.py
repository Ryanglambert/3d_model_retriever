import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import binvox_rw_py as binvox


def plot_vox(path):
    with open(path, 'rb') as f:
        model = binvox.read_as_3d_array(f)

    # model.data is a 3D boolean array in this case. Probably could also be float array.
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(45, 135)
    ax.voxels(model.data, edgecolor='k')
    plt.show()


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
