import SimpleITK as sitk
import numpy as np


def dilate_sitk(labels, RAD):
    slabels = sitk.GetImageFromArray(labels)
    gd = sitk.GrayscaleDilateImageFilter()
    gd.SetKernelRadius(RAD)
    return sitk.GetArrayFromImage(gd.Execute(slabels))


def calc_mask_exclude_overlap(nuclabel, RINGWIDTH=5):
    """
    Examples:
        >>> template = np.zeros((5, 5))
        >>> template[1, 1] = 1
        >>> template[-2, -2] = 2
        >>> calc_mask_exclude_overlap(template, 2)
        array([[False, False, False, False, False],
               [False, False,  True,  True, False],
               [False,  True,  True,  True, False],
               [False,  True,  True, False, False],
               [False, False, False, False, False]], dtype=bool)
    """
    dilated_nuc = dilate_sitk(nuclabel.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 6e4 - nuclabel
    comp_dilated_nuc[comp_dilated_nuc == 6e4] = 0
    comp_dilated_nuc = dilate_sitk(comp_dilated_nuc.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 6e4 - comp_dilated_nuc
    comp_dilated_nuc[comp_dilated_nuc == 6e4] = 0
    mask = comp_dilated_nuc != dilated_nuc
    return mask


def dilate_to_cytoring(labels, RINGWIDTH, MARGIN):
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
    dilated_nuc = dilate_sitk(labels.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 1e4 - labels
    comp_dilated_nuc[comp_dilated_nuc == 1e4] = 0
    comp_dilated_nuc = dilate_sitk(comp_dilated_nuc.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 1e4 - comp_dilated_nuc
    comp_dilated_nuc[comp_dilated_nuc == 1e4] = 0
    dilated_nuc[comp_dilated_nuc != dilated_nuc] = 0
    if MARGIN == 0:
        antinucmask = labels
    else:
        antinucmask = dilate_sitk(np.int32(labels), MARGIN)
    dilated_nuc[antinucmask.astype(bool)] = 0
    return dilated_nuc.astype(np.uint16)


def dilate_to_cytoring_buffer(labels, RINGWIDTH, MARGIN, BUFFER):
    """
    Examples:
        >>> template = np.zeros((5, 5))
        >>> template[1, 1] = 1
        >>> template[-2, -2] = 2
        >>> dilate_to_cytoring_buffer(template, 2, 0, 1)
        array([[1, 1, 0, 0, 0],
               [1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 2],
               [0, 0, 0, 2, 2]], dtype=uint16)
    """

    dilated_nuc = dilate_sitk(labels.astype(np.int32), RINGWIDTH)
    mask = calc_mask_exclude_overlap(labels, RINGWIDTH+BUFFER)
    comp_dilated_nuc = 1e4 - labels
    comp_dilated_nuc[comp_dilated_nuc == 1e4] = 0
    comp_dilated_nuc = dilate_sitk(comp_dilated_nuc.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 1e4 - comp_dilated_nuc
    comp_dilated_nuc[comp_dilated_nuc == 1e4] = 0
    dilated_nuc[comp_dilated_nuc != dilated_nuc] = 0
    if MARGIN == 0:
        antinucmask = labels
    else:
        antinucmask = dilate_sitk(np.int32(labels), MARGIN)
    dilated_nuc[antinucmask.astype(bool)] = 0
    dilated_nuc[mask] = 0
    return dilated_nuc.astype(np.uint16)