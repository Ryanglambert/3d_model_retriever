import argparse
import os
import subprocess
import sys

if 'linux' in sys.platform:
    CMD = ("xvfb-run -s '-screen 0 640x480x24'" 
           "./binvox -cb -pb -e -c -d {dim} {path}")
elif 'darwin' in sys.platform:
    CMD = './binvox -cb -e -c -d {dim} {path}'
else:
    raise SystemError('System is not of type darwin or linux, what kind of system are you using?')



def convert_off_to_binvox(path, dim=30):
    "Converts *.off file to *.binvox"
    assert ".off" in path, 'Wrong file type!'

    process = subprocess.Popen(CMD.format(dim=dim, path=path).split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        print("Output from Binvox: \n{}".format(output))
        raise Exception("Binvox didn't do something right\nBinvox Error:\n{}".format(error))


def list_file_paths(path):
    ignore_files = ['.DS_Store']
    for root, _, files in os.walk(path):
        for file_path in files:
            if file_path not in ignore_files:
                abs_path = os.path.abspath(os.path.join(root, file_path))
                yield abs_path
                print(abs_path)


def main():
    parser = argparse.ArgumentParser(description='Process *.off files into *.binvox')
    parser.add_argument(dest="root_path",
                        help="give the root_path for the *.off files you want to convert to *.binvox")
    parser.add_argument('--dimensions', default=30)
    args = parser.parse_args()

    paths_generator = list_file_paths(args.root_path)
    for path in paths_generator:
        convert_off_to_binvox(path, dim=args.dimensions)


if __name__ == '__main__':
    main()
