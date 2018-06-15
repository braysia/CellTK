# tf_deepcell

#### A simple training and prediction
```
python train.py -i data/nuc0.png -l data/labels0.tif -m data/tests_model.py -o output -n 3000 -e 2 -p 61
python predict.py -i data/nuc1.png -w output/cnn_model_weights.hdf5 -m data/tests_model.py -o output
```

#### Resume training
```
python train.py -i data/nuc0.png -l data/labels0.tif -m data/weights.tests.hdf5
```

#### Using multiple channels
```
python train.py -i data/FIXME -l data/FIXME -m data/tests_model.py
```

#### Training using multiple images
```
python train.py -i data/nuc0.png / data/nuc0.png -l data/labels0.tif / data/labels1.tif -m data/tests_model.py
```

#### Prediction example
The following hdf5 was trained with ```-n 1500000 -e 10 -p 61 -b 256``` with GPU.
```
python predict.py -i data/nuc1.png -w data/tests_pretrained.hdf5 -m data/tests_model.py -o output
```

Use tensorflow (1.3.0) and Cuda 8.0
