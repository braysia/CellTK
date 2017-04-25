"""
Any operations to make img from img.

python celltk/preprocess.py -f gaussian_laplace -i c0/img_00000000*
"""


# from scipy.ndimage import imread
import argparse
import tifffile as tiff
from os.path import basename, join
import numpy as np
import os
import preprocess_operation
import ast
from utils.global_holder import holder
from utils.file_io import make_dirs
from utils.util import imread
from utils.parser import ParamParser


def parse_image_files(inputs):
    if "/" not in inputs:
        return inputs
    store = []
    li = []
    while inputs:
        element = inputs.pop(0)
        if element == "/":
            store.append(li)
            li = []
        else:
            li.append(element)
    store.append(li)
    return zip(*store)


def imsave(img, output, path):
    if isinstance(path, list) or isinstance(path, tuple):
        for num, p in enumerate(path):
            tiff.imsave(join(output, basename(p)), img[:, :, num].astype(np.float32))
    else:
        tiff.imsave(join(output, basename(path)), img.astype(np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="images", nargs="*")
    parser.add_argument("-o", "--output", help="output directory", type=str, default='temp')
    parser.add_argument("-f", "--functions", help="functions", nargs="*")
    parser.add_argument("-p", "--param", nargs="*", help="parameters", default={})
    args = parser.parse_args()
    make_dirs(args.output)

    params = ParamParser(args.param).run()
    args.input = parse_image_files(args.input)
    holder.args = args

    for holder.frame, path in enumerate(args.input):
        img = imread(path)
        for function, param in zip(args.functions, params):
            func = getattr(preprocess_operation, function)
            img = func(img, **param)
        print holder.frame
        imsave(img, args.output, path)


if __name__ == "__main__":
    main()
