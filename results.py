import numpy as np
import os

from keras.models import Model


RESULTS_PATH = 'results/'
FILTER_PLOTS = 'filter_plots/'
PRECISION_RECALL_PLOTS = 'precision_recall_plots/'
TSNE_PLOTS = 'tsne_plots/'
LOGS = 'logs/'


def _initialize_dir(name):
    try:
        os.mkdir(name)
    except FileExistsError:
        print("Directory exists and that's ok let's continue")


def initialize_results_dir(model_name):
    base_path = os.path.join(RESULTS_PATH, model_name)
    _initialize_dir(base_path)
    _initialize_dir(os.path.join(base_path, FILTER_PLOTS))
    _initialize_dir(os.path.join(base_path,
                                 PRECISION_RECALL_PLOTS))
    _initialize_dir(os.path.join(base_path, TSNE_PLOTS))
    _initialize_dir(os.path.join(base_path, LOGS))


def _save_model(model, model_name: str, accuracy: str):
    model_name = model_name + '_' + accuracy
    base_path = os.path.join(RESULTS_PATH, model_name)
    model.save(base_path)


def _make_tsne_plots(eval_model):
    return None


def _make_precision_recall(eval_model):
    return None


def _make_filter_plots(manipulate_model):
    return None


def _make_latent_space(model, x):
    return model.predict(x)


def _make_latent_model(model, layer=-3)
    return Model(model.input, model.layers[layer].output)


def _store_latent_model(model, x, layer=-3):


def _accuracy(eval_model, x_test, y_test):
    y_pred, x_recon = eval_model.predict(x_test)
    test_accuracy = np.rum(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
    return test_accuracy


def process_results(train_model, eval_model, manipulate_model, x_test, y_test, **callbacks):
    _make_tsne_plots(eval_model)
    _make_precision_recall(eval_model)
    _make_filter_plots(manipulate_model)
    accuracy = _accuracy(eval_model, x_test, y_test)
    return None

