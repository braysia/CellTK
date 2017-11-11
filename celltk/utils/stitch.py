from os.path import join
import os
import glob
import numpy as np
import tifffile as tiff
from utils.preprocess_utils import histogram_matching

""" Stitch images with Fiji results
Exported from Stitch_image_Grid_Sequence in Fiji

points = [(0.0, 0.0), (-556.46246, -8.537842), (-565.82874, 533.8285), (-131.09012, 483.0899)]
dataDir=['output/stitch_test/Pos7_n4','output/stitch_test/Pos8_n4',\
         'output/stitch_test/Pos9_n4','output/stitch_test/Pos10_n4']

"""

def relative_position(points):
    points_flat = [int(el) for inner in points for el in inner]
    xpoints = points_flat[1::2]
    ypoints = points_flat[0::2]
    max_xp = min(xpoints)
    max_yp = min(ypoints)
    rel_xpoints = map(lambda l: l - max_xp, xpoints)
    rel_ypoints = map(lambda l: l - max_yp, ypoints)
    rel_points = (rel_xpoints, rel_ypoints)
    return rel_points

def replace_peri(img, val=0):
    img[0, :] = val
    img[-1, :] = val
    img[:, 0] = val
    img[:, -1] = val
    return img

def stitch_images(dataDir, rel_points, output_path):
    # set image params
    file_list=glob.glob(join(dataDir[0],'*.tif'))
    img=tiff.imread(join(os.getcwd(), file_list[0]))
    fn_el=file_list[0].split("_") # file name element
    fp_list = []
    tmp_list = set([fn.split('_')[-3] for fn in file_list]) #fp list
    for x in tmp_list:
        fp_list.append(x)
    t_list = [fn.split('_')[-1] for fn in file_list]  # timepoint list
    imShape=img.shape
    rel_xpoints=rel_points[0]
    rel_ypoints=rel_points[1]

    for fp in fp_list:
        for l in t_list:
            # initialize
            i = 0
            stitchImg = np.zeros([max(rel_xpoints) + imShape[0], max(rel_ypoints) + imShape[1], len(dataDir)])

            for path in dataDir:
                file_list = glob.glob(join(path, '*'+fp+'*.tif'))
                target_file = [h for h in file_list if l in h][0]
                # print target_file
                try:
                    tmp = tiff.imread(join(os.getcwd(), target_file))
                    tmp = histogram_matching(tmp, tmp2, BINS=100, QUANT=2, THRES=False)
                    tmp2 = tmp.copy()
                except:
                    tmp = tiff.imread(join(os.getcwd(), target_file))
                    tmp2 = tmp.copy()
                tmp = replace_peri(tmp)
                stitchImg[rel_xpoints[i]:rel_xpoints[i]+imShape[0], rel_ypoints[i]:rel_ypoints[i]+imShape[1],\
                dataDir.index(path)] = tmp
                stitchImg2 = np.array(stitchImg.max(axis=2), dtype=np.float32)
                i = i + 1

        tiff.imsave(output_path+'stitch' + fn_el[-3] + '_' + l, stitchImg2.astype(stitchImg2.dtype))