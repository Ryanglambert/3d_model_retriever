import os
import subprocess


def convert_off_to_binvox(path, dim=30):
    "Converts *.off file to *.binvox"
    assert ".off" in path, 'Wrong file type!'

    cmd = './binvox -cb -d {dim} {path}'.format(dim=dim, path=path)
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        print("Output from Binvox: \n{}".format(output))
        raise Exception("Binvox didn't do something right\nBinvox Error:\n{}".format(error))


def list_file_paths(path):
    ignore_files = ['.DS_Store']
    for root, _, files in os.walk(path):
        for file_path in files:
            if file_path not in ignore_files:
                print(os.path.abspath(os.path.join(root, file_path)))
                convert_off_to_binvox(file_path)
