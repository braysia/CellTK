"""
python celltk/subdetect.py -l c3/img_00000000* -f ring_dilation -o c4 -p MARGIN=0
"""

from scipy.ndimage import imread
import tifffile as tiff
import argparse
from os.path import basename, join
import numpy as np
import subdetect_operation
from itertools import izip_longest
from utils.file_io import make_dirs
from utils.parser import ParamParser
from utils.global_holder import holder


def caller(inputs, inputs_labels, output, functions, params):
    img = None
    for path, pathl in izip_longest(inputs, inputs_labels):
        if path is not None:
            img = imread(path)
        labels0 = tiff.imread(pathl).astype(np.int16)
        for function, param in zip(functions, params):
            func = getattr(subdetect_operation, function)
            if img is not None:
                labels = func(labels0, img, **param)
            else:
                labels = func(labels0, **param)
        tiff.imsave(join(output, basename(pathl)), labels.astype(np.int16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="images", nargs="*", default=[])
    parser.add_argument("-l", "--labels", help="labels", nargs="+", default=[])
    parser.add_argument("-o", "--output", help="output directory",
                        type=str, default='temp')
    parser.add_argument("-f", "--functions", help="functions", nargs="+")
    parser.add_argument("-p", "--param", nargs="*", help="parameters", default=[])
    args = parser.parse_args()
    make_dirs(args.output)

    params = ParamParser(args.param).run()
    holder.args = args

    caller(args.input, args.labels, args.output, args.functions, params)

if __name__ == "__main__":
    main()
