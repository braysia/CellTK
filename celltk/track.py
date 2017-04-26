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
from utils.file_io import make_dirs
from utils.parser import ParamParser
from utils.global_holder import holder


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
    # parser.add_argument("-p", "--param", nargs="*", help="parameters", type=lambda kv: kv.split("="), default={})
    parser.add_argument("-p", "--param", nargs="*", help="parameters", default=[])
    args = parser.parse_args()
    make_dirs(args.output)
    holder.args = args

    params = ParamParser(args.param).run()

    img0, labels0 = imread(args.input[0]), tiff.imread(args.labels[0]).astype(np.int16)
    tiff.imsave(join(args.output, basename(args.input[0])), labels0.astype(np.int16))
    for holder.frame, (path, pathl) in enumerate(zip(args.input[1:], args.labels[1:])):
        img1, labels1 = imread(path), tiff.imread(pathl).astype(np.int16)
        labels1 = -labels1
        for fnum, (function, param) in enumerate(zip(args.functions, params)):
            func = getattr(track_operation, function)
            if not (labels1 < 0).any():
                continue
            labels0, labels1 = func(img0, img1, labels0, -labels1, **param)

        print holder.frame, len(np.unique(labels1[labels1>0]))
        labels0 = neg2poslabels(labels1)
        img0 = img1
        tiff.imsave(join(args.output, basename(path)), labels0.astype(np.int16))

if __name__ == "__main__":
    main()
