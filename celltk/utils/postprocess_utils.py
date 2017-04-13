from skimage.measure._regionprops import _RegionProperties
import numpy as np
import scipy.ndimage as ndi


class _RegionProperties2(_RegionProperties):
    parent = None
    next = None

    @property
    def total_intensity(self):
        return np.sum(self.intensity_image[self.image])


def regionprops(label_image, intensity_image=None, cache=True):
    label_image = np.squeeze(label_image)

    if label_image.ndim not in (2, 3):
        raise TypeError('Only 2-D and 3-D images supported.')

    if not np.issubdtype(label_image.dtype, np.integer):
        raise TypeError('Label image must be of integral type.')

    regions = []

    objects = ndi.find_objects(label_image)
    for i, sl in enumerate(objects):
        if sl is None:
            continue

        label = i + 1

        props = _RegionProperties2(sl, label, label_image, intensity_image,
                                  cache)
        regions.append(props)

    return regions
