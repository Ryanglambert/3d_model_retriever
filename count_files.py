from IPython.display import display, HTML
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


def plot_unbalanced_classes(df, title):
    grouped = df.groupby(['class', 'type'])['type'].count()
    unstacked = grouped.unstack()
    plt.figure()
    unstacked = unstacked.sort_values(['test', 'train'])
    unstacked.plot(kind='bar', title=title, figsize=(10, 6), yticks=np.arange(0, 900, 100))
    plt.ylabel('Number of Samples')
    plt.grid()
    plt.show()
    display(unstacked)
