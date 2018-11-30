# CellTK

Live-cell analysis toolkit.

Image processing is simply an image conversion/transformation process.
CellTK has the following five major processes which all implement conversion between img and labels.

1. preprocessing:   img -> img
2. segmentation:   img -> labels
3. subdetection: labels (and img) -> labels
4. tracking: labels -> labels*
5. postprocessing: labels -> labels*

where
- img: np.ndarray[np.float32] (e.g. a raw image from a microscope)
- labels: np.ndarray[np.int16] (e.g. nuclear objects)
\* tracked objects have consistent values over frames

For each processes, you can find a module named ___\*\_operation.py___. (e.g. _celltk/preprocess_operations.py_).

These files are the "repositories" of functions.
They simply contain a list of functions which takes an input and convert images. If you need a new function, simply add it to here.


When you input a raw image, it should take TIFF or PNG files with various datatypes as well.

### Command line Example:
The simplest way to apply a function is to use ___command.py___.
This option is convenient to play with functions and parameters.


```
python celltk/command.py -i data/testimages0/CFP/img* -f constant_thres -p THRES=2000 -o output/c1
python celltk/command.py -i data/testimages0/CFP/img* -l output/c1/img* -f run_lap track_neck_cut -o output/nuc
```

___-i___ for images path, ___-l___ for labels path, ___-o___ for an output directory, ___-f___ for a function name from ___*operation.py___ modules, ___-p___ for arguments to the function.

Note that, time-lapse files need to have file names in a sorted order.


### Caller Example:
You can run a pipeline of operations using ___celltk/caller.py___.

```
python celltk/caller.py input_files/input_tests1.yml
```

This configuration file contains operations defined like this:
```
- function: constant_thres
  images: /example/img_00*.tif
  output: output_0
  params:
    THRES: 500
```

You can find how to set up a configuration file [here](doc/CONFIGURE_YML.md).

### Apply to extract single-cell properties
After segmenting and tracking cells, we want to extract single-cell properties as a table.

Unlike other five major processes, ___celltk/apply.py___ produces __csv__ and __npz__ file as an output.

```
python celltk/apply.py -i data/testimages0/CFP/img* -l output/nuc/img* -o output/array.npz
```
By default, it will use a folder name as a table key.
To specify table keys, use ___-p___ and ___-s___ in a command line.
```
python celltk/apply.py -i data/testimages0/YFP/img* -l output/nuc/img* -o output/array.npz -p nuc -s YFP
```

Or use ___obj\_names___ and ___ch\_names___ in a caller.
```
# Sample YML
- function: apply
  images:
    - DAPI/img*
    - TRITC/img*
  labels:
    - op001
    - op002
  ch_names:
    - DAPI
    - JNKKTR
  obj_names:
    - nuc
    - cyto
```


The output can be loaded with LabeledArray class.
e.g.
```
python -c "from celltk.labeledarray import LabeledArray;arr = LabeledArray().load('output/array.npz');print arr.labels;print arr['CFP', 'nuc', 'x']"
```

For visualization and manipulation of these arrays, I recommend to take a loot at [covertrace](https://github.com/braysia/covertrace).


## Install dependencies

Due to the priority issue, install with the following command:
```
cat requirements.txt | xargs -n 1 pip install
```

The other option is to use Docker container.
```
docker pull braysia/celltk
docker run -it -v /$FOLDER_TO_MOUNT:/home/ braysia/celltk
```
Please modify $FOLDER_TO_MOUNT, like `docker run -it -v /Users/kudo/example:/home/ braysia/celltk`.

You can add "-p 8888:8888" for running jupyter notebook from the docker image.

## Covert lab specific details
- [Parallelization using clusters and MongoDB](fireworks/README_FIREWORKS.md)

