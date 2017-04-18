"""
python celltk/segment.py -i ~/covertrack/data/testimages/img_0000000* -f example_thres -o c1 -p THRES=2000
"""


from scipy.ndimage import imread
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


def make_dirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def clean_labels(labels, rad, OPEN=2):
    """default cleaning. Fill holes, remove small and large objects and opening.
    """
    labels = gray_fill_holes(labels)
    labels = clear_border(labels, buffer_size=2)
    labels = remove_small_objects(labels, rad[0]**2 * np.pi, connectivity=4)
    antimask = remove_small_objects(labels, rad[1]**2 * np.pi, connectivity=4)
    labels[antimask] = False
    labels = label(binary_opening(labels, np.ones((int(OPEN), int(OPEN))), iterations=1))
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="images", nargs="*")
    parser.add_argument("-o", "--output", help="output directory",
                        type=str, default='temp')
    parser.add_argument("-f", "--functions", help="functions", nargs="*", default=None)
    parser.add_argument("-p", "--param", nargs="*", help="parameters", type=lambda kv: kv.split("="), default={})
    parser.add_argument("-r", "--radius", help="minimum and maximum radius", nargs=2, default=[3, 50])
    parser.add_argument("--open", help="OPENING parameters", nargs=1, default=2)
    args = parser.parse_args()
    make_dirs(args.output)
    param = dict(args.param)
    for key, value in param.iteritems():
        param[key] = ast.literal_eval(value)

    if args.functions is None:
        print help(segment_operation)

    for path in args.input:
        img = imread(path)
        for function in args.functions:
            func = getattr(segment_operation, function)
            img = func(img, **param)
        labels = clean_labels(img, args.radius, args.open)
        tiff.imsave(join(args.output, basename(path)), labels.astype(np.float32))


if __name__ == "__main__":
    main()
