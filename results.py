import csv
import numpy as np
import os

from itertools import cycle
from keras.models import Model
from sklearn.metrics import (confusion_matrix,
                             precision_recall_curve,
                             average_precision_score)

from plots import plot_confusion_matrix
from utils import (class_subset,
                   query_latent_space,
                   average_precision)

RESULTS_PATH = 'results/'
FILTER_PLOTS = 'filter_plots/'
PRECISION_RECALL_PLOTS = 'precision_recall_plots/'
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
    _initialize_dir(os.path.join(base_path, MODELS))
    return base_path


def _make_latent_space(model, x):
    return model.predict(x)


def _make_latent_model(model, layer=-3):
    return Model(model.input, model.layers[layer].output)


def _get_average_precisions(latent_model, latent_space, x_test, y_test):
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
    return average_precisions


def _make_tsne_plots(eval_model, save_name: str):
    return None


def _make_precision_recall(eval_model):
    return None


def _make_filter_plots(manipulate_model):
    return None


def _save_model_summary(model, path):
    # def myprint(s):
    #     with open(os.path.join(path, 'modelsummary.txt'), 'w') as f:
    #         print(s, file=f)
    with open(os.path.join(path, 'modelsummary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def _accuracy(eval_model, x_test, y_test):
    y_pred, x_recon = eval_model.predict(x_test)
    test_accuracy = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
    return test_accuracy


def _save_details(path, **kwargs):
    file_path = os.path.join(path, 'details.csv')
    with open(file_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in kwargs.items():
            writer.writerow([key, value])


def save_map_plot(average_precisions, path):
    import matplotlib.pyplot as plt
    mean_average_precision = np.mean(average_precisions)
    plt.hist(average_precisions, bins=10)
    plt.text(.1, 500, 'Mean Average Precision: {:.2%}'.format(mean_average_precision))
    plt.vlines(mean_average_precision, 0, 800)
    plt.title('Mean Average Precision')
    plt.savefig(os.path.join(path, 'mean_average_precision.png'), bbox_inches='tight')


def save_tsne_plot(latent_space, path):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from MulticoreTSNE import MulticoreTSNE as TSNE

    tsne = TSNE(3, n_jobs=4)
    reduced = tsne.fit_transform(latent_space)
    fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2])
    ax.view_init(30, 45)
    plt.savefig(os.path.join(path, 'TSNE.png'), bbox_inches='tight')


def save_confusion_matrix(y_test, y_pred, target_names, path, figsize=(15, 15)):
    import matplotlib.pyplot as plt
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    plt.figure(figsize=figsize)
    plot_confusion_matrix(cm, target_names, normalize=True)
    plt.savefig(os.path.join(path, 'Confusion_matrix.png'), bbox_inches='tight')


def plot_precision_recall(y_test, y_pred, target_names,
                          path, save=False, show_figs=False,
                          figsize=(7, 8)):
    import matplotlib.pyplot as plt
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(y_pred.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_pred[:, i])
        average_precision[i] = average_precision_score(y_test[:, i],
                                                       y_pred[:, i])

    # order = sorted(average_precision, key=lambda x: x[1], reverse=True)
    order = list(zip(*sorted(average_precision.items(),
                             key=lambda x: x[1],
                             reverse=True)))[0]

    fig = plt.figure(figsize=figsize)
    lines = []
    labels = []
    fig_count = 0
    count = 0
    # for i, color in zip(range(y_test.shape[1]), colors):
    for idx, i, color in zip(range(1, y_test.shape[1]+1), order, colors):
        l, =plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {} (area= {})'\
                          .format(target_names[i], round(average_precision[i], 2)))
    #     print(round(average_precision[i], 2))
        if idx % 5 == 0 and idx != 0:
    #         fig = plt.gcf()
            fig.subplots_adjust(bottom=0.25)
            plt.legend(lines, labels, loc=(0, .18))
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall')
            lines = []
            labels = []
            if save:
                plt.savefig(os.path.join(path, PRECISION_RECALL_PLOTS, 'precision_recall {}'.format(fig_count)))
                fig_count += 1
            if show_figs:
                plt.show()
            plt.close()
            plt.figure(figsize=figsize)
        count += 1


def process_results(name: str, train_model, eval_model,
                    manipulate_model, x_test, y_test, target_names,
                    **details):
    "Takes all outputs you care about and logs them to results folder"
    latent_model = _make_latent_model(eval_model)
    latent_space = _make_latent_space(latent_model, x_test)
    accuracy = str(round(_accuracy(eval_model,
                                   x_test,
                                   y_test), 5)).replace('.', '')
    average_precisions = _get_average_precisions(latent_model,
                                                 latent_space,
                                                 x_test, y_test)
    mean_avg_prec = str(round(np.mean(average_precisions),5)).replace('.', '')
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
    y_pred, x_recon = eval_model.predict(x_test)

    # save map plots
    save_map_plot(average_precisions, dir_path)
    # save tsne plots
    save_tsne_plot(latent_space, dir_path)
    # save confusion matrix
    save_confusion_matrix(y_test, y_pred, target_names, dir_path)
    # save precision recall plots
    plot_precision_recall(y_test, y_pred, target_names, dir_path, save=True)

