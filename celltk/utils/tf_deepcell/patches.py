from __future__ import division
import numpy as np
import os
try:
    from tensorflow.python.keras import backend
    from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, Iterator, array_to_img
except:
    from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator, Iterator, array_to_img
    from tensorflow.contrib.keras.python.keras import backend as K


def _sample_coords_weighted(num, shape, weights):
    flat_idx = np.arange(shape[0] * shape[1])
    chosen_flat = np.random.choice(flat_idx, num, p=weights/weights.sum())
    return np.unravel_index(chosen_flat, shape)


def _calc_equal_weights(features):
    ap = []
    for i in np.unique(features):
        ap.append((features == i).sum())
    frac = np.array([(np.sum(ap) - i)/np.sum(ap) for i in ap])
    prob_2d = np.zeros(features.shape)
    for i in np.unique(features):
        prob_2d[features == i] = frac[i]
    return prob_2d


def pick_coords(num, features, patch_h, patch_w):
    """
    features: img with labels
    """
    prob_2d = _calc_equal_weights(features.astype(np.uint8))
    _ph, _pw = int(np.floor(patch_h/2)), int(np.floor(patch_w/2))
    perim = np.zeros(prob_2d.shape, dtype=np.bool)
    perim[_ph:-_ph, _pw:-_pw] = True
    prob_2d[~perim] = 0
    return _sample_coords_weighted(num, features.shape, prob_2d.flatten())


def pick_coords_list(num, li_features, patch_h, patch_w):
    """
    features: img with labels
    """
    from random import shuffle
    num = int(num/len(li_features))
    li_coords = []
    for en, features in enumerate(li_features):
        coords = pick_coords(num, features, patch_h, patch_w)
        coords = zip(*(np.ones(num, np.uint8) * en, coords[0], coords[1]))
        li_coords.extend(coords)
    shuffle(li_coords)
    return li_coords



# def _pick_coords(num, features, patch_h, patch_w):
#     """
#     features: img with labels
#     """
#     prob_2d = _calc_equal_weights(features.astype(np.uint8))
#     _ph, _pw = int(np.floor(patch_h/2)), int(np.floor(patch_w/2))
#     perim = np.zeros(prob_2d.shape, dtype=np.bool)
#     perim[_ph:-_ph, _pw:-_pw] = True
#     prob_2d[~perim] = 0
#     return _sample_coords_weighted(num, features.shape, prob_2d.flatten())


def extract_patches(num, x, y, patch_h, patch_w):
    """
    x: input images
    y: feature image with labels

    Sample many windows from a large image. It will correct for labels unbalance.
    (If there are less labels for cell boundaries, it increases the sampling probability)
    """
    coords = pick_coords(num, y, patch_h, patch_w)
    h, w = int(np.floor(patch_h/2)), int(np.floor(patch_w/2))
    xstack = np.zeros((num, patch_h, patch_w, x.shape[-1]), np.float32)
    ystack = np.zeros(num)
    for n, (ch, cw) in enumerate(zip(*coords)):
        xstack[n, :, :] = x[ch-h:ch+h+1, cw-w:cw+w+1]
        ystack[n] = y[ch, cw]
    return xstack, ystack


def _extract_patches(x, y, coords, patch_h, patch_w):
    """
    x: input images x.shape = (hpix, wpix, ch)
    y: feature image with labels

    Sample many windows from a large image. It will correct for labels unbalance.
    (If there are less labels for cell boundaries, it increases the sampling probability)
    """
    h, w = int(np.floor(patch_h/2)), int(np.floor(patch_w/2))
    xstack = np.zeros((len(coords), patch_h, patch_w, x.shape[-1]), np.float32)
    ystack = np.zeros(len(coords))
    for n, (ch, cw) in enumerate(coords):
        xstack[n, :, :, :] = x[ch-h:ch+h+1, cw-w:cw+w+1]
        ystack[n] = y[ch, cw]
    return xstack, ystack


def extract_patch_list(lix, liy, ecoords, patch_h, patch_w):
    h, w = int(np.floor(patch_h/2)), int(np.floor(patch_w/2))
    xstack = np.zeros((len(ecoords), patch_h, patch_w, lix[0].shape[-1]), np.float32)
    ystack = np.zeros(len(ecoords))
    for n, (cn, ch, cw) in enumerate(ecoords):
        xstack[n, :, :, :] = lix[cn][ch-h:ch+h+1, cw-w:cw+w+1]
        ystack[n] = liy[cn][ch, cw]
    return xstack, ystack


class PatchDataGenerator(ImageDataGenerator):
    def flow(self, x, y, coords, patch_h, patch_w, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png'):
        return CropIterator(
            x, y, self, coords=coords, patch_h=patch_h, patch_w=patch_w,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


class CropIterator(Iterator):
    def __init__(self, x, y, image_data_generator, coords, patch_h, patch_w,
                 batch_size=32, shuffle=False, seed=None, func_patch=_extract_patches,
                 data_format=None, save_to_dir=None, save_prefix='', save_format='png'):
        self.coords = coords
        self.patch_h = patch_h
        self.patch_w = patch_w
        if isinstance(x, list):
            self._x, self._y = x[:], y[:]
            chnum = self._x[0].shape[-1]
        else:
            self._x, self._y = x.copy(), y.copy()
            self._x = self._x[0, :, :, :]
            chnum = self._x.shape[-1]
        self.x = np.asarray(np.zeros((1, patch_h, patch_w, chnum)), dtype=K.floatx())
        if y is not None:
            self.y = np.zeros(1)
        else:
            self.y = None
        self.n = len(self.coords)
        self.func_patch = func_patch

        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(CropIterator, self).__init__(len(self.coords), batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]), dtype=K.floatx())
        batch_coords = [self.coords[i] for i in index_array]
        x, y = self.func_patch(self._x, self._y, batch_coords, self.patch_h, self.patch_w)
        self.x = x
        self.y = y
        index_array = np.arange(len(self.y))

        for i, j in enumerate(index_array):
            x = self.x[j]
            x = self.image_data_generator.random_transform(x.astype(K.floatx()))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix, index=j,
                hash=np.random.randint(1e4), format=self.save_format)
            img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        if isinstance(index_array, tuple):
            index_array = index_array[0]
        return self._get_batches_of_transformed_samples(index_array)


class PatchDataGeneratorList(ImageDataGenerator):
    def flow(self, x, y, coords, patch_h, patch_w, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png'):
        x = [i[0] for i in x]
        return CropIterator(
            x, y, self, coords=coords, patch_h=patch_h, patch_w=patch_w,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            data_format=self.data_format, func_patch=extract_patch_list,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

