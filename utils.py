import csv
import math

import matplotlib.pyplot as plt
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
    """Stratified shuffled train val split
    Ensures validation set is representative of the class
    distribution in the given dataset.
    arr : np.array
    y_train : np.array

    Returns
    -------
    x_train : np.array
    y_train : np.array
    x_val : np.array
    y_val : np.array
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(1, test_size=test_size)
    split = [i for i in sss.split(arr, y_train)]
    train_idx, val_idx = split[0]
    return arr[train_idx], y_train[train_idx], arr[val_idx], y_train[val_idx]


def class_subset(x, y, class_name, class_name_list):
    """Gets subsets of x and y based on class_name
    x : np.array
    y : np.array
        dummy variable style (columns for each class)
    class_name : str
    class_name_list : list[str's]
    class_subset(x_test, y_test, 'glass_box', target_names)
    Returns
    -------
    x : np.array
        subset from given x
    y : np.array
        subset from given y
    original_indices : np.array
        the corresponding indices to the original arrays
        so you know where you came from
    """
    class_indices = np.argmax(y, axis=1) == class_name_list.index(class_name)
    original_indices = np.arange(0, x.shape[0])[class_indices]
    return x[class_indices], y[class_indices], original_indices


def query_corpus(latent_object, latent_corpus, query_size=10):
    """Query a latent corpus (bottleneck features from NN)
    latent_object : np.array
        object as represented in an N-D space
    latent_corpus : np.array
        matrix of N-d latent vectors each representing a model
        in your associated corpus
    query_size : int
        number of relevant items to retrieve

    Returns
    -------
    top_n_sorted_sims : np.array
        cosine similarities for the top n rows in latent_corpus
        where n is specified by `query_size`
    top_n_sorted_indices : np.array
        indices of top n rows from the latent_corpus
    """
    sims = latent_corpus.dot(latent_object.T)
    sorted_sims_indices = np.argpartition(sims,
                                          range(-query_size, 0),
                                          axis=0)
    top_n_sorted_indices = sorted_sims_indices[:-query_size-1:-1].ravel()
    top_n_sorted_sims = sims[top_n_sorted_indices].ravel()
    return top_n_sorted_sims, top_n_sorted_indices


def upsample_classes(arr, y_train):
    """Naively upsamples to the class with the highest count
    Balancing all the classes in your training set

    arr : np.array
    classes: np.array

    Returns
    -------
    arr : np.array
        upsampled array balancing based on y_train array
    y_train : np.array
        labels associated with the upsampled arr
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


def plot_log(filename, show=True):
    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))
        values[:,0] += 1

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for i, key in enumerate(keys):
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()
