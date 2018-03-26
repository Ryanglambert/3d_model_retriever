import csv
import numpy as np
import os

from keras.models import Model

from utils import (class_subset,
                   query_latent_space,
                   average_precision)

RESULTS_PATH = 'results/'
FILTER_PLOTS = 'filter_plots/'
PRECISION_RECALL_PLOTS = 'precision_recall_plots/'
TSNE_PLOTS = 'tsne_plots/'
MODELS = 'models/'


def _initialize_dir(name):
    try:
        os.mkdir(name)
    except FileExistsError:
        print("Directory exists and that's ok let's continue")


def initialize_results_dir(model_name, accuracy, mean_average_precision):
    model_dir = "{}_acc_{}_map_{}".format(model_name, accuracy, mean_average_precision)
    base_path = os.path.join(RESULTS_PATH, model_dir)
    _initialize_dir(base_path)
    _initialize_dir(os.path.join(base_path, FILTER_PLOTS))
    _initialize_dir(os.path.join(base_path,
                                 PRECISION_RECALL_PLOTS))
    _initialize_dir(os.path.join(base_path, TSNE_PLOTS))
    _initialize_dir(os.path.join(base_path, MODELS))
    return base_path


def _make_latent_space(model, x):
    return model.predict(x)


def _make_latent_model(model, layer=-3):
    return Model(model.input, model.layers[layer].output)


def _mean_average_precision(latent_model, latent_space, x_test, y_test):
    """Calculate mean precision"""
    average_precisions = np.zeros(x_test.shape[0])
    for i in range(x_test.shape[0]):
        if i % 100 == 0:
            print('precisions_done_calculating{}'.format(i))
        num = i
        num_retrievable = (np.argmax(y_test[num]) == \
                               np.argmax(y_test, axis=1)).sum()
        latent_object = latent_model.predict(x_test[num:num+1])
        sims, latent_indices = query_latent_space(latent_object,
                                                  latent_space,
                                                  x_test.shape[0])
        ranked_relevant = np.argmax(y_test[num]) ==\
                            np.argmax(y_test[latent_indices], axis=1)

        average_precisions[i] = average_precision(ranked_relevant, num_retrievable)

    mean_average_precision = np.mean(average_precisions)
    return mean_average_precision


def _make_tsne_plots(eval_model, save_name: str):
    return None


def _make_precision_recall(eval_model):
    return None


def _make_filter_plots(manipulate_model):
    return None


def _save_model_summary(model, path):
    def myprint(s):
        with open(os.path.join(path, 'modelsummary.txt'), 'w') as f:
            print(s, file=f)
    model.summary(print_fn=myprint)


def _accuracy(eval_model, x_test, y_test):
    y_pred, x_recon = eval_model.predict(x_test)
    test_accuracy = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
    return test_accuracy


def _save_details(path, **kwargs):
    file_path = os.path.join(path, 'details.txt')
    with open(file_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in kwargs.items():
            writer.writerow([key, value])


def process_results(name: str, train_model, eval_model,
                    manipulate_model, x_test, y_test, **details):
    latent_model = _make_latent_model(eval_model)
    latent_space = _make_latent_space(latent_model, x_test)
    accuracy = str(round(_accuracy(eval_model,
                                   x_test,
                                   y_test), 5)).replace('.', '')
    mean_avg_prec = str(round(_mean_average_precision(latent_model,
                                                      latent_space,
                                                      x_test, y_test),5)).replace('.', '')
    dir_path = initialize_results_dir(name, accuracy, mean_avg_prec)
    _save_details(dir_path, **details)

    # latent space and model
    latent_model.save(os.path.join(dir_path, MODELS, 'latent_model.hdf5'))
    np.save(os.path.join(dir_path, 'latent_space.npy'), latent_space)
    # all the other models hdf5 files
    train_model.save(os.path.join(dir_path, MODELS, 'train_model.hdf5'))
    # only saving text of train_model since that contains everything we need to know
    _save_model_summary(train_model, dir_path)
    eval_model.save(os.path.join(dir_path, MODELS, 'eval_model.hdf5'))
    manipulate_model.save(os.path.join(dir_path, MODELS, 'manipulate_model.hdf5'))

    # save tsne plots
        # also as gifs
    # save precision recall plots
    # save filter plots

    # _make_tsne_plots(eval_model)
    # _make_precision_recall(eval_model)
    # _make_filter_plots(manipulate_model)

