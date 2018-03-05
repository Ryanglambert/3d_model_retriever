import numpy as np


def partition_classes(x_train, y_train_labels):
    "separates x_train and y_train_labels into class partitions"
    unique_y = np.unique(y_train_labels)
    class_list = []
    for i in unique_y:
        class_list.append((i, x_train[y_train_labels == i]))
    return class_list 
