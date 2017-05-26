
import imp
import argparse
import types
from segment import caller as segment_caller
import importlib
from os.path import join, isdir, exists
import preprocess, segment, track, postprocess, subdetect, apply
import preprocess_operation, segment_operation, track_operation, postprocess_operation, subdetect_operation
from glob import glob
from joblib import Parallel, delayed
import logging
import yaml

logger = logging.getLogger(__name__)


ops_modules = [preprocess_operation, segment_operation, track_operation, postprocess_operation, subdetect_operation, apply]
caller_modules = [preprocess, segment, track, postprocess, subdetect, apply]


def extract_path(path):
    f = glob(path)
    if isdir(f[0]) or not f:
        f = glob(join(path, '*'))
    return f


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
            in0 = zip(*[extract_path(join(outputdir, i)) for i in inputs])
        # if not in0:
        #     in0 = zip(*[glob(join(outputdir, i, '*')) for i in inputs])
    return in0



def retrieve_in_list(obj, key, empty=[]):
    if isinstance(obj, dict):
        obj = [obj, ]
    st = []
    for ob in obj:
        if key not in ob:
            st.append(empty)
        else:
            st.append(ob[key])
    return st


def parse_operation(operation):
    functions = retrieve_in_list(operation, 'function')
    params = retrieve_in_list(operation, 'params', empty={})
    images = retrieve_in_list(operation, 'images')[0]
    labels = retrieve_in_list(operation, 'labels')[0]
    output = retrieve_in_list(operation, 'output')[-1]
    return functions, params, images, labels, output


def _retrieve_caller_based_on_function(function):
    module = [m for m, top in zip(caller_modules, ops_modules) if hasattr(top, function)][0]
    return getattr(module, "caller")


def run_operation(output_dir, operation):
    functions, params, images, labels, output = parse_operation(operation)
    inputs = prepare_path_list(images, output_dir)
    inputs_labels = prepare_path_list(labels, output_dir)
    output = join(output_dir, output) if output else output_dir
    caller = _retrieve_caller_based_on_function(functions[0])

    if len(functions) == 1 and functions[0] == 'apply':
        ch_names = operation['ch_names'] if 'ch_names' in operation else images
        obj_names = operation['obj_names'] if 'obj_names' in operation else labels
        caller(inputs, inputs_labels, output, obj_names, ch_names)
    elif not inputs_labels:
        caller(inputs, output, functions, params=params)
    else:
        caller(inputs, inputs_labels, output, functions, params=params)


def run_operations(output_dir, operations):
    for operation in operations:
        run_operation(output_dir, operation)


def load_yaml(path):
    with open(path) as stream:
        contents = yaml.load(stream)
    return contents


def single_call(inputs):
    contents = load_yaml(inputs)
    logging.basicConfig(level=logging.INFO)
    run_operations(contents['OUTPUT_DIR'], contents['operations'])
    logger.info("Caller finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--cores", help="number of cores for multiprocessing",
                        type=int, default=1)
    parser.add_argument("input", nargs="*", help="input argument file path")
    args = parser.parse_args()

    if len(args.input) == 1:
        single_call(args.input[0])
    if len(args.input) > 1:
        num_cores = args.cores
        print str(num_cores) + ' started parallel'
        Parallel(n_jobs=num_cores)(delayed(single_call)(i) for i in args.input)


if __name__ == "__main__":
    main()
