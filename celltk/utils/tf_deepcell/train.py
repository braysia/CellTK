from __future__ import division, print_function
import os
import numpy as np
try:
    from tensorflow.python.keras import optimizers, callbacks
    from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
except:
    from tensorflow.contrib.keras import optimizers, callbacks
    from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
from tfutils import imread
from patches import extract_patches, pick_coords, pick_coords_list, extract_patch_list, _extract_patches, PatchDataGeneratorList
from tfutils import load_model_py, make_outputdir
from os.path import join
from tfutils import parse_image_files

FRAC_TEST = 0.1


def define_callbacks(output, batch_size):
    csv_logger = callbacks.CSVLogger(join(output, 'training.log'))
    earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=2)
    tensorboard = callbacks.TensorBoard(batch_size=batch_size)
    fpath = join(output, 'weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5')
    cp_cb = callbacks.ModelCheckpoint(filepath=fpath, monitor='val_loss', save_best_only=True)
    return [csv_logger, earlystop, tensorboard, cp_cb]


def train(image_list, labels_list, model_path, output, patchsize=61, nsamples=10000,
          batch_size=32, nepochs=100, frac_test=FRAC_TEST):
    assert np.bool(patchsize & 0x1)  # check if odd
    model = load_model_py(model_path)
    model.summary()

    li_image, li_labels = [], []
    for image_path, labels_path in zip(image_list, labels_list):
        image, labels = imread(image_path), imread(labels_path).astype(np.uint8)
        if image.ndim == 2:
            image = np.expand_dims(image, -1)
        elif image.ndim == 3:
            image = np.moveaxis(image, 0, -1)
        li_image.append(image)
        li_labels.append(labels)

    num_tests = int(nsamples * FRAC_TEST)
    ecoords = pick_coords_list(nsamples, li_labels, patchsize, patchsize)
    ecoords_tests, ecoords_train = ecoords[:num_tests], ecoords[num_tests:],
    x_tests, y_tests = extract_patch_list(li_image, li_labels, ecoords_tests, patchsize, patchsize)
    li_image = [np.expand_dims(i, 0) for i in li_image]

    make_outputdir(output)
    opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    callbacksets = define_callbacks(output, batch_size)

    datagen = PatchDataGeneratorList(rotation_range=90, shear_range=0,
                                     horizontal_flip=True, vertical_flip=True)
    history = model.fit_generator(datagen.flow(li_image, li_labels, ecoords_train, patchsize, patchsize, batch_size=batch_size, shuffle=True),
                                  steps_per_epoch=len(ecoords_train)/batch_size,
                                  epochs=nepochs,
                                  validation_data=(x_tests, y_tests),
                                  validation_steps=len(ecoords_train)/batch_size,
                                  callbacks=callbacksets)

    score = model.evaluate(x_tests, y_tests, batch_size=batch_size)
    print('score[loss, accuracy]:', score)
    rec = dict(acc=history.history['acc'], val_acc=history.history['val_acc'],
               loss=history.history['loss'], val_loss=history.history['val_loss'])
    np.savez(join(output, 'records.npz'), **rec)

    json_string = model.to_json()
    open(join(output, 'cnn_model.json'), 'w').write(json_string)
    model.save_weights(join(output, 'cnn_model_weights.hdf5'))
    yaml_string = model.to_yaml()
    open(join(output, 'cnn_model.yaml'), 'w').write(yaml_string)


def _parse_command_line_args():
    """
    image:  Path to a tif or png file (e.g. data/nuc0.png).
            To pass multiple image files (size can be varied), use syntax like
            "-i im0.tif / im1.tif / im2.tif", and pass the same number of labels.
    labels: (e.g. data/labels0.tif)
    model:  path to a python file describing a model (e.g. data/tests_model.py)
            or weights.*.hdf5 produced by ModelCheckPoint.
            To resume training, pass the hdf5 file.
    n:      A number of pixels for training. Use a large number (like 1,000,000)
    batch:  Typically 128-512?
    """

    import argparse
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('-i', '--image', help='image file path', nargs="*")
    parser.add_argument('-l', '--labels', help='labels file path', nargs="*")
    parser.add_argument('-m', '--model', help='path to python file or hdf5')
    parser.add_argument('-o', '--output', default='.', help='output directory')
    parser.add_argument('-n', '--nsamples', type=int, default=10000, help='number of samples')
    parser.add_argument('-b', '--batch', type=int, default=64)
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-p', '--patch', type=int, default=61,
                        help='pixel size of image patches. make it odd')
    return parser.parse_args()


def _main():
    args = _parse_command_line_args()
    images = parse_image_files(args.image)[0]
    labels = parse_image_files(args.labels)[0]
    train(images, labels, args.model, args.output, args.patch,
          args.nsamples, args.batch, args.epoch)


if __name__ == "__main__":
    _main()
