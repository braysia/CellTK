import SimpleITK as sitk
import numpy as np


def dilate_sitk(label, RAD):
    slabel = sitk.GetImageFromArray(label)
    gd = sitk.GrayscaleDilateImageFilter()
    gd.SetKernelRadius(RAD)
    return sitk.GetArrayFromImage(gd.Execute(slabel))


def dilate_to_cytoring(label, RINGWIDTH, MARGIN):
    """
    Examples:
        >>> template = np.zeros((5, 5))
        >>> template[2, 2] = 1
        >>> dilate_to_cytoring(template, 1, 0)
        array([[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 0, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0]], dtype=uint16)
        >>> dilate_to_cytoring(template, 2, 1)
        array([[0, 1, 1, 1, 0],
               [1, 0, 0, 0, 1],
               [1, 0, 0, 0, 1],
               [1, 0, 0, 0, 1],
               [0, 1, 1, 1, 0]], dtype=uint16)
    """
    dilated_nuc = dilate_sitk(label.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 1e4 - label
    comp_dilated_nuc[comp_dilated_nuc == 1e4] = 0
    comp_dilated_nuc = dilate_sitk(comp_dilated_nuc.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 1e4 - comp_dilated_nuc
    comp_dilated_nuc[comp_dilated_nuc == 1e4] = 0
    dilated_nuc[comp_dilated_nuc != dilated_nuc] = 0
    if MARGIN == 0:
        antinucmask = label
    else:
        antinucmask = dilate_sitk(np.int32(label), MARGIN)
    dilated_nuc[antinucmask.astype(bool)] = 0
    return dilated_nuc.astype(np.uint16)
