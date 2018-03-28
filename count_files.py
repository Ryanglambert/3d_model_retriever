from IPython.display import display, HTML
import matplotlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

skip_files = ['.DS_Store', 'README.txt', '__MACOSX']


def make_file_description_data_frame(path='./ModelNet10/', show_missed=True):
    paths = []
    for root, dirs, files in os.walk(path):
        path = root.split(os.sep)
        for file in files:
            class_type = path[2]
            try:
                # This could be better, but this isn't very important right now
                # handling when the path terminates at a directory instead of a file
                # So we're not acctually missing any files from being counted
                sample_type = path[3]
            except:
                if show_missed:
                    print("woops!:")
                    print(path)
                break
            paths.append({
                'class': class_type,
                'type': sample_type,
                'file': file
            }) if not file in skip_files else None
    return pd.DataFrame(paths)


def plot_unbalanced_classes(df, title, display_df=False, save=False, figsize=(10, 6), fontsize=10):
    grouped = df.groupby(['class', 'type'])['type'].count()
    unstacked = grouped.unstack()
    plt.figure(figsize=figsize)
    unstacked = unstacked.sort_values(['test', 'train'])
    default_font_size = matplotlib.rcParams['font.size']
    matplotlib.rcParams.update({'font.size': fontsize})
    unstacked.plot(kind='bar', figsize=figsize, yticks=np.arange(0, unstacked['train'].max(), 500))
    matplotlib.rcParams.update({'font.size': default_font_size})
    plt.title(title)
    plt.ylabel('Number of Samples')
    # plt.grid()
    if save:
        plt.savefig(title + '.png')
    plt.show()
    if display_df:
        display(unstacked)
