"""
python celltk/segmentation.py -i ~/covertrack/data/testimages/img_0000000* -f examp
le_thres -o c1 THRES=2000
"""


from scipy.ndimage import imread
import argparse
import tifffile as tiff
from os.path import basename, join
import numpy as np
import os
import segment_operation
import ast


def make_dirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="images", nargs="*")
    parser.add_argument("-o", "--output", help="output directory",
                        type=str, default='temp')
    parser.add_argument("-f", "--functions", help="functions", nargs="*")
    parser.add_argument("param", nargs="*", help="input argument file path", type=lambda kv: kv.split("="))
    args = parser.parse_args()
    make_dirs(args.output)
    param = dict(args.param)
    for key, value in param.iteritems():
        param[key] = ast.literal_eval(value)

    for path in args.input:
        img = imread(path)
        for function in args.functions:
            func = getattr(segment_operation, function)
            img = func(img, **param)
        tiff.imsave(join(args.output, basename(path)), img.astype(np.float32))

if __name__ == "__main__":
    main()
