# CellTK


Example:
```
python celltk/segment.py -i data/testimages0/CFP/img* -f constant_thres -p THRES=2000 -o output/c1

python celltk/track.py -i data/testimages0/CFP/img* -l output/c1/img* -f run_lap track_neck_cut -o output/nuc

python celltk/postprocess.py -i data/testimages0/CFP/img* -l output/c2/img* -f gap_closing -o output/nuc

python celltk/subdetect.py -l output/nuc/img* -f ring_dilation -o output/cyto

python celltk/apply.py -i data/testimages0/CFP/img* -l output/nuc/img* -o output/array.npz

python celltk/apply.py -i data/testimages0/YFP/img* -l output/nuc/img* -o output/array.npz

python celltk/apply.py -i data/testimages0/YFP/img* -l output/cyto/img* -o output/array.npz
```

The output can be loaded with LabeledArray class.
```
python -c "from celltk.labeledarray import LabeledArray;arr = LabeledArray().load('output/array.npz');print arr.labels;print arr['CFP', 'nuc', 'x']"
```


## Running Docker
```
docker pull braysia/celltk
docker run -it -v /folder_you_want_to_mount:/home/ braysia/celltk
```
Add "-p 8888:8888" for running jupyter notebook from the docker image.
