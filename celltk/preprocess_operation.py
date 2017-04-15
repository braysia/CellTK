from __future__ import division
from utils.filters import calc_lapgauss
import SimpleITK as sitk
import numpy as np


def gaussian_laplace(img, SIGMA=2.5):
    return calc_lapgauss(img, SIGMA)


def curvature_anisotropic_smooth(img, NUMITER=10):
    fil = sitk.CurvatureAnisotropicDiffusionImageFilter()
    fil.SetNumberOfIterations(NUMITER)
    simg = sitk.GetImageFromArray(img.astype(np.float32))
    sres = fil.Execute(simg)
    return sitk.GetArrayFromImage(sres)