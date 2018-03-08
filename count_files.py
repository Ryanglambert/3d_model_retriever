import os
import pandas as pd

skip_files = ['.DS_Store', 'README.txt', '__MACOSX']


def make_file_description_data_frame():
    paths = []
    for root, dirs, files in os.walk("./ModelNet10/"):
        path = root.split(os.sep)
        for file in files:
            class_type = path[2]
            try:
                # This could be better, but this isn't very important right now
                sample_type = path[3]
            except:
                print("woops!:")
                print(path)
                break
            paths.append({
                'class': class_type,
                'type': sample_type,
                'file': file
            }) if not file in skip_files else None
    return pd.DataFrame(paths)