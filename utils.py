import numpy as np


def get_class_dist(y_train):
    "Return a class distribution for an array"
    classes = np.unique(y_train)
    classes = np.sort(classes)
    class_dist = np.zeros(classes.shape[0]).astype(int)
    for class_num in classes:
        class_dist[class_num] = y_train[y_train == class_num].shape[0]
    return classes, class_dist


def get_upsample_amount(class_counts):
    most_samples = class_counts[np.argmax(class_counts)]
    num_upsample = most_samples - class_counts
    return num_upsample


def get_indices_by_class(arr):
    return None


def upsample(*arrs, n=0):
    "sample with replacement"
    # make sure same dim
    for i, arr in enumerate(arrs[1::2]):
        assert arrs[i].shape[0] == arrs[i+1].shape[0], "first dim must all be the same"
    # make indices
    indices = np.random.randint(0, arrs[0].shape[0], n)
    # do sampling
    new_arrs = []
    for arr in arrs:
        new_arrs.append(arr[indices])
    return new_arrs
