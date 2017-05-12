
import imp
import argparse
import types
from segment import caller as segment_caller
import importlib
from os.path import join, isdir, exists
import preprocess, segment, track, postprocess, subdetect, apply
import preprocess_operation, segment_operation, track_operation, postprocess_operation, subdetect_operation
from glob import glob

operations = [preprocess_operation, segment_operation, track_operation, postprocess_operation, subdetect_operation, apply]
caller_modules = [preprocess, segment, track, postprocess, subdetect, apply]


class CallerParser(object):
    def __init__(self, argfile):
        self.argfile = argfile
        self.argdict = {}

    def set_explicit_args(self):
        ia_args = [a for a in dir(self.argfile) if not a.startswith('_')]
        ia_args = [a for a in ia_args if not isinstance(getattr(self.argfile, a), types.ModuleType)]
        ia_args = [a for a in ia_args if not isinstance(getattr(self.argfile, a), types.FunctionType)]
        for a in ia_args:
            self.argdict[a] = getattr(self.argfile, a)


def prepare_path_list(inputs, outputdir):
    if isinstance(inputs, str):
        in0 = glob(inputs)
        if not in0:
            in0 = glob(join(outputdir, inputs))
        if isdir(in0[0]):
            in0 = glob(join(in0[0], '*'))
    elif isinstance(inputs, list):
        if all([exists(i) for i in inputs]):
            return inputs
        in0 = zip(*[glob(i) for i in inputs])
        if not in0:
            in0 = zip(*[glob(join(i, '*')) for i in inputs])
        if not in0:
            in0 = zip(*[glob(join(outputdir, i)) for i in inputs])
        if not in0:
            in0 = zip(*[glob(join(outputdir, i, '*')) for i in inputs])
    return in0


def run_operation(argdict):
    inputs, inputs_labels = [], []
    ops = sorted([i for i in argdict.iterkeys() if i.startswith("op")])
    for op in ops:
        methods = argdict[op]
        methods = [methods, ] if not isinstance(methods, list) else methods

        inputdir = [i for i in methods if 'inputdir' in i]
        if inputdir:
            inputs = inputdir[0].pop('inputdir')
        inputs = prepare_path_list(inputs, argdict['OUTPUT_DIR'])

        labels_folder = [i for i in methods if 'labels_folder' in i]
        if labels_folder:
            inputs_labels = labels_folder[0].pop("labels_folder")
        inputs_labels = prepare_path_list(inputs_labels, argdict['OUTPUT_DIR'])

        functions, params = [], []
        for method in methods:

            if "output_folder" not in method:
                output = join(argdict['OUTPUT_DIR'], op)
            else:
                output = join(argdict['OUTPUT_DIR'], method.pop('output_folder'))
            functions.append(method.pop('function'))
            params.append(method)

        module = [m for m, top in zip(caller_modules, operations) if hasattr(top, functions[0])][0]
        caller = getattr(module, "caller")

        if functions[0] == 'apply':
            ch_folders = method.pop('ch_folders')
            obj_folders = method.pop('obj_folders')
            inputs_list = [prepare_path_list(ch, argdict['OUTPUT_DIR']) for ch in ch_folders]
            inputs_labels_list = [prepare_path_list(obj, argdict['OUTPUT_DIR']) for obj in obj_folders]
            ch_names = ch_folders if 'ch_names' not in method else method.pop('ch_names')
            obj_names = obj_folders if 'obj_names' not in method else method.pop('obj_names')
            caller(inputs_list, inputs_labels_list, argdict['OUTPUT_DIR'], obj_names, ch_names)
            return
        if 'preprocess' in str(module) or 'segment' in str(module):
            caller(inputs, output, functions, params=params)
        else:
            caller(inputs, inputs_labels, output, functions, params=params)

        if module == preprocess:
            inputs = output
        else:
            inputs_labels = output

from joblib import Parallel, delayed


def single_call(inputs):
    argfile = imp.load_source('inputArgs', inputs)
    cp = CallerParser(argfile)
    cp.set_explicit_args()
    argdict = cp.argdict
    run_operation(argdict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="*", help="input argument file path")
    args = parser.parse_args()

    if len(args.input) == 1:
        single_call(args.input[0])
    if len(args.input) > 1:
        num_cores = 4
        print str(num_cores) + ' started parallel'
        Parallel(n_jobs=num_cores)(delayed(single_call)(i) for i in args.input)


if __name__ == "__main__":
    main()
