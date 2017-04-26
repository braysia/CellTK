"""
python celltk/subdetect.py -l c3/img_00000000* -f ring_dilation -o c4 -p MARGIN=0
"""

from scipy.ndimage import imread
import tifffile as tiff
import argparse
from os.path import basename, join
import numpy as np
import os
import subdetect_operation
import ast
from itertools import izip_longest
from utils.file_io import make_dirs
from utils.parser import ParamParser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", help="images", nargs="*", default=[])
    parser.add_argument("-l", "--labels", help="labels", nargs="+", default=[])
    parser.add_argument("-o", "--output", help="output directory",
                        type=str, default='temp')
    parser.add_argument("-f", "--functions", help="functions", nargs="+")
    parser.add_argument("-p", "--param", nargs="*", help="parameters", default=[])
    args = parser.parse_args()
    make_dirs(args.output)

    params = ParamParser(args.param).run()

    img = None
    for path, pathl in izip_longest(args.images, args.labels):
        if path is not None:
            img = imread(path)
        print pathl
        labels0 = tiff.imread(pathl).astype(np.int16)
        for function, param in zip(args.functions, params):
            func = getattr(subdetect_operation, function)
            if img is not None:
                labels = func(labels0, img, **param)
            else:
                labels = func(labels0, **param)
        tiff.imsave(join(args.output, basename(pathl)), labels.astype(np.int16))


if __name__ == "__main__":
    main()
