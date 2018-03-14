import numpy as np


def get_class_dist(y_train):
    "Return a class distribution for an array"
    classes = np.unique(y_train)
    classes = np.sort(classes)
    class_dist = np.zeros(classes.shape[0]).astype(int)
    for class_num in classes:
        class_dist[class_num] = y_train[y_train == class_num].shape[0]
    return classes, class_dist


def upsample_classes(arr, y_train):
    """
    Upsamples samples based on class distribution

    Args:
        arr: Numpy array of shape (n, p)
        classes: Numpy array of shape (n,)

    Return:
        arr: Numpy array of shape (m, p) where m is now bigger than n
    """
    indices = np.indices(y_train.shape).reshape(-1)
    classes, class_counts = get_class_dist(y_train)
    upsample_amount = np.max(class_counts)
    class_skipped = classes[np.argmax(class_counts)]

    total_new_samples = upsample_amount * classes.shape[0]
    new_arr = np.zeros((total_new_samples,) + arr.shape[1:])
    new_y_train = np.zeros(total_new_samples)
    for nth_class in classes:
        nth_indices = indices[y_train == nth_class]
        if nth_class == class_skipped:
            new_arr[nth_indices] = arr[nth_indices]
            new_y_train[nth_indices] = y_train[nth_indices]
        else:
            sample_indices = np.random.choice(nth_indices, upsample_amount)
            new_arr[sample_indices] = arr[sample_indices]
            new_y_train[sample_indices] = y_train[sample_indices]
    return new_arr, new_y_train
