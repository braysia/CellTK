# CellTK

Caller Example:
```
celltk input_files/ktr_inputs/input_anisoinh.yml
```


Commandline Example:
```
python celltk/segment.py -i data/testimages0/CFP/img* -f constant_thres -p THRES=2000 -o output/c1

python celltk/track.py -i data/testimages0/CFP/img* -l output/c1/img* -f run_lap track_neck_cut -o output/nuc

python celltk/postprocess.py -i data/testimages0/CFP/img* -l output/c2/img* -f gap_closing -o output/nuc

python celltk/subdetect.py -l output/nuc/img* -f ring_dilation -o output/cyto

python celltk/apply.py -i data/testimages0/CFP/img* -l output/nuc/img* -o output/array.npz

python celltk/apply.py -i data/testimages0/YFP/img* -l output/nuc/img* -o output/array.npz

python celltk/apply.py -i data/testimages0/YFP/img* -l output/cyto/img* -o output/array.npz
```
_-i_ for images path, _-l_ for labels path, _-o_ for output directory, _-f_ for function name from *operation.py, _-p_ for arguments to the function.

The output can be loaded with LabeledArray class. e.g.
```
python -c "from celltk.labeledarray import LabeledArray;arr = LabeledArray().load('output/array.npz');print arr.labels;print arr['CFP', 'nuc', 'x']"
```


## Running Docker
```
docker pull braysia/celltk
docker run -it -v /folder_you_want_to_mount:/home/ braysia/celltk
```
Add "-p 8888:8888" for running jupyter notebook from the docker image.

## Processes
Currently there are five major processes.
1. preprocess
2. segment
3. subdetect
4. tracking
5. postprocess

For each processes, you can find two modules in celltk (e.g. preprocess.py and preprocess_operations.py). 
The *\*_operations.py* file contains a list of functions, which they take an input image and transform it. 

You can quickly check functions available by typing the following commands:
```
celltk-preprocess
celltk-segment
celltk-subdetect
celltk-tracking
celltk-postprocess
```


Two major data types recurrently used are "img" and "labels".  
img: np.ndarray[np.float32]  
labels: np.ndarray[np.int16]

Each processes have an input and output of a certain data type.  
1. preprocess: img -> img
2. segment: img -> labels
3. subdetect: labels (and img) -> labels
4. tracking: labels -> labels (where tracked objects have the same value over time)
5. postprocess: labels -> labels (where tracked objects have the same value over time)



