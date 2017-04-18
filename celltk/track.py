"""
python celltk/tracking.py -f nearest_neighbor -i c0/img_00000000* -l c1/img_0000000
0*  -p DISPLACEMENT=10 MASSTHRES=0.2
"""

from scipy.ndimage import imread
import argparse
import tifffile as tiff
from os.path import basename, join
import numpy as np
import os
import track_operation
import ast


def make_dirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def neg2poslabels(labels):
    maxint = labels.max()
    negatives = np.unique(labels[labels < 0])
    for i in negatives:
        maxint += 1
        labels[labels == i] = maxint
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="images", nargs="+")
    parser.add_argument("-l", "--labels", help="labels", nargs="+")
    parser.add_argument("-o", "--output", help="output directory",
                        type=str, default='temp')
    parser.add_argument("-f", "--functions", help="functions", nargs="+")
    parser.add_argument("-p", "--param", nargs="*", help="parameters", type=lambda kv: kv.split("="), default={})
    args = parser.parse_args()
    make_dirs(args.output)
    param = dict(args.param)
    for key, value in param.iteritems():
        param[key] = ast.literal_eval(value)

    img0, labels0 = imread(args.input[0]), imread(args.labels[0]).astype(np.int32)
    tiff.imsave(join(args.output, basename(args.input[0])), labels0.astype(np.int32))
    for path, pathl in zip(args.input[1:], args.labels[1:]):
        img1, labels1 = imread(path), imread(pathl).astype(np.int32)
        labels1 = -labels1
        for fnum, function in enumerate(args.functions):
            func = getattr(track_operation, function)
            if not (labels1 < 0).any():
                continue
            labels0, labels1 = func(img0, img1, labels0, -labels1, **param)
        labels0 = neg2poslabels(labels1)
        img0 = img1
        tiff.imsave(join(args.output, basename(path)), labels0.astype(np.int32))


if __name__ == "__main__":
    main()
