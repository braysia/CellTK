"""
python celltk/subdetect.py -l c3/img_00000000* -f ring_dilation -o c4 -p MARGIN=0
"""

from scipy.ndimage import imread
import argparse
import tifffile as tiff
from os.path import basename, join, dirname
import numpy as np
import os
import subdetect_operation
import ast
from utils.postprocess_utils import regionprops # set default parent and next as None
from LabeledArray.labeledarray.labeledarray import LabeledArray


def make_dirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def add_parent_id(labels, img, cells):
    return cells


cells = regionprops(labels, img)
cells = add_parent_id(labels, img, cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", help="images", nargs="*", default=[])
    parser.add_argument("-l", "--labels", help="labels", nargs="+")
    parser.add_argument("-o", "--output", help="output directory",
                        type=str, default='temp')
    # parser.add_argument("-f", "--functions", help="functions", nargs="+")
    # parser.add_argument("-p", "--param", nargs="*", help="parameters", type=lambda kv: kv.split("="))
    args = parser.parse_args()
    make_dirs(args.output)
    param = dict(args.param)
    for key, value in param.iteritems():
        param[key] = ast.literal_eval(value)

    ch_name, obj_name = basename(dirname(args.images[0])), basename(dirname(args.labels[0])), 

    for path, pathl in zip(args.images, args.labels):
        img, labels = imread(path), imread(pathl).astype(np.int32)
        # for function in args.functions:
        #     func = getattr(subdetect_operation, function)
        #     if img is not None:
        #         labels = func(labels0, img, **param)
        #     else:
        #         labels = func(labels0, **param)
        tiff.imsave(join(args.output, basename(pathl)), labels.astype(np.int32))


if __name__ == "__main__":
    main()
