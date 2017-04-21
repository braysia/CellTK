from __future__ import division
from utils.filters import calc_lapgauss
import SimpleITK as sitk
import numpy as np
from utils.preprocess_utils import homogenize_intensity_n4
from utils.preprocess_utils import convert_positive, estimate_background_prc
from utils.preprocess_utils import curvature_anisotropic_smooth, resize_img
from utils.preprocess_utils import histogram_matching, wavelet_subtraction_hazen
from utils.filters import adaptive_thresh


def gaussian_laplace(img, SIGMA=2.5, NEG=False):
    if NEG:
        return -calc_lapgauss(img, SIGMA)
    return calc_lapgauss(img, SIGMA)


def curvature_anisotropic_smooth(img, NUMITER=10):
    fil = sitk.CurvatureAnisotropicDiffusionImageFilter()
    fil.SetNumberOfIterations(NUMITER)
    simg = sitk.GetImageFromArray(img.astype(np.float32))
    sres = fil.Execute(simg)
    return sitk.GetArrayFromImage(sres)


def smooth_curvature_anisotropic(img, holder, NUMITER=10):
    """anisotropic diffusion on a scalar using the modified curvature diffusion equation (MCDE).
    """
    return curvature_anisotropic_smooth(img, NUMITER)


def background_subtraction_wavelet_hazen(img, holder, THRES=100, ITER=5, WLEVEL=6, OFFSET=50):
    """Wavelet background subtraction.
    """
    back = wavelet_subtraction_hazen(img, ITER=ITER, THRES=THRES, WLEVEL=WLEVEL)
    img = img - back
    return convert_positive(img, OFFSET)


def n4_illum_correction(img, holder, RATIO=1.5, FILTERINGSIZE=50):
    """
    Implementation of the N4 bias field correction algorithm.
    Takes some calculation time. It first calculates the background using adaptive_thesh.
    """
    bw = adaptive_thresh(img, RATIO=RATIO, FILTERINGSIZE=FILTERINGSIZE)
    img = homogenize_intensity_n4(img, -bw)
    return img


def n4_illum_correction_downsample(img, holder, DOWN=2, RATIO=1.05, FILTERINGSIZE=50, OFFSET=10):
    """Faster but more insensitive to local illum bias.
    """
    fil = sitk.ShrinkImageFilter()
    cc = sitk.GetArrayFromImage(fil.Execute(sitk.GetImageFromArray(img), [DOWN, DOWN]))
    bw = adaptive_thresh(cc, RATIO=RATIO, FILTERINGSIZE=FILTERINGSIZE/DOWN)
    himg = homogenize_intensity_n4(cc, -bw)
    himg = cc - himg
    # himg[himg < 0] = 0
    bias = resize_img(himg, img.shape)
    img = img - bias
    return convert_positive(img, OFFSET)


# def flatfield_with_inputs(img, ff_paths=['img00.tif', 'img01.tif']):
#     """
#     Examples:
#         preprocess_args = (dict(name='flatfield_with_inputs', ch="TRITC", ff_paths=['Exp2_w1TRITC_s1_t1.TIF', 'Exp2_w1TRITC_s1_t1.TIF']), )
#     """
#     ff_store = []
#     for path in ff_paths:
#         ff_store.append(imread(path))
#         ff = np.median(np.dstack(ff_store), axis=2)
#     img = img - ff
#     img[img < 0] = 0
#     return img
