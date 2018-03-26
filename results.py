import os


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

