from __future__ import division
import os
import numpy as np
from os.path import join, basename, splitext
from tfutils import imread
try:
    from tensorflow.python.keras import backend
except:
    from tensorflow.contrib.keras.python.keras import backend
from tfutils import convert_model_patch2full, load_model_py, make_outputdir
import tifffile as tiff


def predict(img_path, model_path, weight_path):
    x = imread(img_path)

    if x.ndim == 2:
        x = np.expand_dims(x, -1)
    elif x.ndim == 3:
        x = np.moveaxis(x, 0, -1)
    x = np.expand_dims(x, 0)

    model = load_model_py(model_path)
    model = convert_model_patch2full(model)
    model.load_weights(weight_path)

    model.summary()
    evaluate_model = backend.function(
        [model.layers[0].input, backend.learning_phase()],
        [model.layers[-1].output]
        )

    cc = evaluate_model([x, 0])[0]

    # from tensorflow.contrib.keras import optimizers
    # opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # cc = model.predict(x)
    return [cc[0, :, :, i] for i in range(cc.shape[-1])]


def save_output(outputdir, images, pattern):
    make_outputdir(outputdir)
    for num, img in enumerate(images):
        tiff.imsave(join(outputdir, '{0}_l{1}.tif'.format(pattern, num)), img)


def _parse_command_line_args():
    import argparse
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('-i', '--image', help='image file path')
    parser.add_argument('-w', '--weight', help='hdf5 file path')
    parser.add_argument('-m', '--model', help='python file path with models')
    parser.add_argument('-o', '--output', default='.', help='output directory')
    return parser.parse_args()


def _main():
    args = _parse_command_line_args()
    images = predict(args.image, args.model, args.weight)
    save_output(args.output, images, splitext(basename(args.image))[0])


if __name__ == "__main__":
    _main()
