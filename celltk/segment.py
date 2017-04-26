"""
python celltk/segment.py -i ~/covertrack/data/testimages/img_0000000* -f example_thres -o c1 -p THRES=2000
"""


# from scipy.ndimage import imread
import argparse
import tifffile as tiff
from os.path import basename, join
import numpy as np
import os
import segment_operation
import ast
from skimage.segmentation import clear_border
from utils.filters import gray_fill_holes
from skimage.morphology import remove_small_objects
from utils.filters import label
from scipy.ndimage.morphology import binary_opening
from utils.util import imread
from utils.file_io import make_dirs
from utils.parser import ParamParser
from utils.global_holder import holder


def clean_labels(labels, rad, OPEN=2):
    """default cleaning. Fill holes, remove small and large objects and opening.
    """
    labels = gray_fill_holes(labels)
    labels = clear_border(labels, buffer_size=2)
    labels = remove_small_objects(labels, rad[0]**2 * np.pi, connectivity=4)
    antimask = remove_small_objects(labels, rad[1]**2 * np.pi, connectivity=4)
    labels[antimask > 0] = False
    labels = label(binary_opening(labels, np.ones((int(OPEN), int(OPEN))), iterations=1))
    return labels


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="images", nargs="*")
    parser.add_argument("-o", "--output", help="output directory",
                        type=str, default='temp')
    parser.add_argument("-f", "--functions", help="functions", nargs="*", default=None)
    parser.add_argument("-p", "--param", nargs="*", help="parameters", default=[])
    parser.add_argument("-r", "--radius", help="minimum and maximum radius", nargs=2, default=[3, 50])
    parser.add_argument("--open", help="OPENING parameters", nargs=1, default=2)
    args = parser.parse_args()
    make_dirs(args.output)

    params = ParamParser(args.param).run()

    if args.functions is None:
        print help(segment_operation)

    holder.args = args
    inputs = parse_image_files(args.input)
    for path in inputs:
        img = imread(path)
        for function, param in zip(args.functions, params):
            func = getattr(segment_operation, function)
            img = func(img, **param)
        if isinstance(path, list) or isinstance(path, tuple):
            path = path[0]
        labels = clean_labels(img, args.radius, args.open)
        tiff.imsave(join(args.output, basename(path)), labels.astype(np.int16))


if __name__ == "__main__":
    main()
