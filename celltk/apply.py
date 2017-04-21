"""
python celltk/apply.py -i c0/img_00000000* -l c2/img_00000000*

TODO:
need to deal with parent id
"""

from scipy.ndimage import imread
import argparse
import tifffile as tiff
from os.path import basename, join, dirname, abspath
import numpy as np
import os
from utils.postprocess_utils import regionprops # set default parent and next as None
from labeledarray import LabeledArray
from os.path import exists


PROP_SAVE = ['area', 'cell_id', 'convex_area', 'cv_intensity',
             'eccentricity', 'major_axis_length', 'minor_axis_length', 'max_intensity',
             'mean_intensity', 'median_intensity', 'min_intensity', 'orientation',
             'perimeter', 'solidity', 'std_intensity', 'total_intensity', 'x', 'y']


def make_dirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def add_parent_id(labels, img, cells):
    return cells


def initialize_arr(store):
    nframe = len(store)
    cell_ids = [[i.cell_id for i in cells] for cells in store]
    cell_ids = [i for j in cell_ids for i in j]
    ncells = len(np.unique(cell_ids))
    return np.zeros((len(PROP_SAVE), ncells, nframe))


def make_labeledarray(store, obj_name, ch_name):
    arr = initialize_arr(store)
    for frame, cells in enumerate(store):
        for cell in cells:
            for pn, prop in enumerate(PROP_SAVE):
                arr[pn, cell.cell_id-1, frame] = getattr(cell, prop)

    label_list = []
    for prop in PROP_SAVE:
        label_list.append([ch_name, obj_name, prop])
    return LabeledArray(arr, label_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", help="images", nargs="*", default=[])
    parser.add_argument("-l", "--labels", help="labels", nargs="+")
    parser.add_argument("-o", "--output", help="file name", type=str, default='temp')
    # parser.add_argument("-f", "--functions", help="functions", nargs="+")
    # parser.add_argument("-p", "--param", nargs="*", help="parameters", type=lambda kv: kv.split("="))
    args = parser.parse_args()

    output = args.output + '.npz' if not args.output.endswith('.npz') else args.output
    make_dirs(dirname(abspath(output)))
    # param = dict(args.param)
    # for key, value in param.iteritems():
    #     param[key] = ast.literal_eval(value)

    ch_name, obj_name = basename(dirname(args.images[0])), basename(dirname(args.labels[0])), 

    store = []
    for path, pathl in zip(args.images, args.labels):
        img, labels = imread(path), imread(pathl).astype(np.int32)
        store.append(regionprops(labels, img))

    larr = make_labeledarray(store, obj_name, ch_name)
    if exists(output):
        ex_larr = LabeledArray().load(output)
        larr = larr.vstack(ex_larr)
    larr.save(output)


if __name__ == "__main__":
    main()
