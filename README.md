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
These files simply contain a list of functions which takes an input and convert images.

#### Command line Example:
The simplest way to apply a function is to use ___command.py___.

```
python celltk/command.py -i data/testimages0/CFP/img* -f constant_thres -p THRES=2000 -o output/c1

python celltk/command.py -i data/testimages0/CFP/img* -l output/c1/img* -f run_lap track_neck_cut -o output/nuc
```

___-i___ for images path, ___-l___ for labels path, ___-o___ for output directory, ___-f___ for function name from ___*operation.py___, ___-p___ for arguments to the function.


#### Caller Example:
```
python celltk/caller.py input_files/ktr_inputs/input_anisoinh.yml
```
The output can be loaded with LabeledArray class. e.g.
```
python -c "from celltk.labeledarray import LabeledArray;arr = LabeledArray().load('output/array.npz');print arr.labels;print arr['CFP', 'nuc', 'x']"
```

## Running Docker Container
```
docker pull braysia/celltk
docker run -it -v /$FOLDER_TO_MOUNT:/home/ braysia/celltk
```
Please modify $FOLDER_TO_MOUNT, like `docker run -it -v /Users/kudo/example:/home/ braysia/celltk`.

You can add "-p 8888:8888" for running jupyter notebook from the docker image.



