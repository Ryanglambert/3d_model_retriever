import pandas as pd
import os

base_dir = 'results'


def get_acc(path):
    _, mnet_path = os.path.split(path)
    acc = mnet_path.split('acc_')[1].split('_map')[0]
    return int(acc) / 10**(len(acc) - 1)


def get_map(path):
    _, mnet_path = os.path.split(path)
    mean_avg_prec = mnet_path.split('map_')[1]
    return int(mean_avg_prec) / 10**(len(mean_avg_prec) - 1)


def load_details_df(model_folder_path):
    df = pd.read_csv(os.path.join(
        model_folder_path, 'details.csv'), header=None).T
    # make header
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    acc = get_acc(model_folder_path)
    map = get_map(model_folder_path)
    df['accuracy'] = acc
    df['mean_avg_prec'] = map
    df['model_path'] = model_folder_path
    return df


def list_results_dir(model):
    return [os.path.join(base_dir, x)
            for x in os.listdir(base_dir)
            if model in x]


def convert_num_to_percent(num):
    str_num = str(num).split('.')[0]
    return num / 10 ** (len(str_num))


def load_results(model='ModelNet10'):
    mnet_path = list_results_dir(model)
    dfs = [load_details_df(mdir) for mdir in mnet_path]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df.columns[::-1]].sort_values(['accuracy', 'mean_avg_prec'], ascending=False)
    first_columns = ['accuracy', 'mean_avg_prec']
    last_columns = ['model_path']
    def column_checker(col_name):
        if col_name in df.columns:
            df[col_name] = df[col_name].map(convert_num_to_percent)
            first_columns.append(col_name)
    column_checker('rot_accuracy')
    column_checker('rot_mean_avg_prec')
    # if 'rot_accuracy' in df.columns:
    #     df['rot_accuracy'] = df['rot_accuracy'].map(convert_num_to_percent)
    #     first_columns.append('rot_accuracy')
    # if 'rot_mean_avg_prec' in df.columns:
    #     df['rot_mean_avg_prec'] = df['rot_mean_avg_prec'].map(convert_num_to_percent)
    #     first_columns.append('rot_mean_avg_prec')
    last_columns = ['model_path']
    not_first_columns = [i for i in df.columns if i not in first_columns and i not in last_columns]
    df = df[first_columns + not_first_columns + last_columns]
    return df

