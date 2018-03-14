import numpy as np


def get_class_dist(y_train):
    "Return a class distribution for an array"
    classes = np.unique(y_train)
    classes = np.sort(classes)
    class_dist = np.zeros(classes.shape[0]).astype(int)
    for class_num in classes:
        class_dist[class_num] = y_train[y_train == class_num].shape[0]
    return classes, class_dist


def stratified_shuffle(arr, y_train, test_size=0.2):
    "Stratified and shuffled sample of classes"
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(1, test_size=test_size)
    split = [i for i in sss.split(arr, y_train)]
    train_idx, val_idx = split[0]
    return arr[train_idx], y_train[train_idx], arr[val_idx], y_train[val_idx]


def upsample_classes(arr, y_train):
    """
    Naively upsamples to the class with the highest count
    Balancing all the classes

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

    # the highest class doesn't get upsampled
    new_arr = arr[y_train == class_skipped]
    new_y_train = y_train[y_train == class_skipped]

    # upsample and append each class to the two new arrays that will be returned
    for nth_class in classes:
        # print('nth_class is: {}'.format(nth_class))
        nth_indices = indices[y_train == nth_class]
        if nth_class != class_skipped:
            sample_indices = np.random.choice(nth_indices, upsample_amount)
            new_arr = np.append(new_arr, arr[sample_indices], axis=0)
            new_y_train = np.append(new_y_train, y_train[sample_indices], axis=0)
    return new_arr.astype(int), new_y_train.astype(int)
