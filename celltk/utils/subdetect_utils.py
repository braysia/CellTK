import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import watershed as skiwatershed
from skimage.measure import label
from skimage.feature import peak_local_max
from skimage.morphology import disk
from skimage.morphology import closing
from skimage.measure import label as skim_label
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d
from skimage.measure import regionprops


def dilate_sitk(labels, RAD):
    slabels = sitk.GetImageFromArray(labels)
    gd = sitk.GrayscaleDilateImageFilter()
    gd.SetKernelRadius(RAD)
    return sitk.GetArrayFromImage(gd.Execute(slabels))


def voronoi_expand(labels, return_line=False):
    dist = distance_transform_edt(labels)

    vor = skiwatershed(-dist, markers=labels)
    if not return_line:
        return vor
    else:
        mask = skiwatershed(-dist, markers=labels, watershed_line=True)
        lines = mask == 0
        return vor, lines


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
    comp_dilated_nuc = 1e4 - labels
    comp_dilated_nuc[comp_dilated_nuc == 1e4] = 0
    comp_dilated_nuc = dilate_sitk(comp_dilated_nuc.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 1e4 - comp_dilated_nuc
    comp_dilated_nuc[comp_dilated_nuc == 1e4] = 0

    vor, vlines = voronoi_expand(labels, return_line=True)
    vor0, vor1 = vor.copy(), vor.copy()
    vor0[comp_dilated_nuc != vor0.copy()] = False
    vor1[dilated_nuc != vor1.copy()] = False
    dilated_nuc = np.max(np.dstack((vor0, vor1)), axis=2)

    if MARGIN == 0:
        antinucmask = labels
    else:
        antinucmask = dilate_sitk(np.int32(labels), MARGIN)
    dilated_nuc[antinucmask.astype(bool)] = 0

    if BUFFER:
        vlines = dilate_sitk(vlines.astype(np.uint32), BUFFER)
        dilated_nuc[vlines > 0] = 0
    return dilated_nuc.astype(np.uint16)


def watershed_labels(labels, regmax):
    # Since there are non-unique values for dist, add very small numbers. This will separate each marker by regmax at least.
    dist = distance_transform_edt(labels) + np.random.rand(*labels.shape)*1e-10
    labeled_maxima = label(peak_local_max(dist, min_distance=int(regmax), indices=False))
    wshed = -dist
    wshed = wshed - np.min(dist)
    markers = np.zeros(wshed.shape, np.int16)
    markers[labeled_maxima > 0] = -labeled_maxima[labeled_maxima > 0]
    markers[labels == 0] = 0
    wlabels = skiwatershed(wshed, markers, connectivity=np.ones((3,3), bool), mask=labels!=0)
    wlabels = -wlabels
    wlabels = labels.max() + wlabels
    wlabels[wlabels == labels.max()] = 0
    all_labels = label(labels + wlabels)
    return all_labels

def pairwise_distance(loc1, loc2):
    xprev = [i[0] for i in loc1]
    yprev = [i[1] for i in loc1]
    xcurr = [i[0] for i in loc2]
    ycurr = [i[1] for i in loc2]
    xprevTile = np.tile(xprev, (len(xcurr), 1))
    yprevTile = np.tile(yprev, (len(ycurr), 1))
    return abs(xprevTile.T - xcurr) + abs(yprevTile.T - ycurr)

def skilabel(bw, conn=2):
    '''original label might label any objects at top left as 1. To get around this pad it first.'''
    bw = np.pad(bw, pad_width=1, mode='constant', constant_values=False) ## ORIGINAL
    #bw = np.pad(bw, pad_width=0, mode='constant', constant_values=False) ## DDED KB
    label = skim_label(bw, connectivity=conn)
    label = label[1:-1, 1:-1]
    return label

def calc_high_pass_kernel(slen, SIGMA):
    """For Salmonella"""
    temp = np.zeros((slen, slen))
    temp[int(slen/2), int(slen/2)] = 1
    gf = gaussian_filter(temp, SIGMA)
    norm = np.ones((slen, slen))/(slen**2)
    return gf - norm

def calc_high_pass(img, slen, SIGMA):
    """For Salmonella"""
    kernel = calc_high_pass_kernel(slen, SIGMA)
    cc = convolve2d(img, kernel, mode='same')
    return cc

def label_high_pass(img, slen=3, SIGMA=0.5, THRES=50, CLOSE=3):
    """For Salmonella"""
    cc = calc_high_pass(img, slen, SIGMA)
    cc[cc < 0] = 0
    la = skilabel(cc > THRES, conn=1) ## KATIE 1/25/19
    la = closing(la, disk(CLOSE))
    return la


def label_nearest(img, label, nuc, DISTHRES=25):
    """Label objects to the nearest nuc.
    """
    nuc_prop = regionprops(nuc, img, cache=False) #extract all region props from nuc seg
    sal_prop = regionprops(label, img, cache=False) #extract all region props from bacteria sec
    nuc_loc = [i.centroid for i in regionprops(nuc, img, cache=False)] # pull centroid for each nucleus
    sal_loc = [i.centroid for i in regionprops(label, img, cache=False)] # pull centrodi for each bac label 
    dist = pairwise_distance(nuc_loc, sal_loc) # calculate distance between nucleus and e coli 

    min_dist_arg = np.argmin(dist, axis=0) 

    template = np.zeros(img.shape, np.uint16)
    for num, (idx, sal) in enumerate(zip(min_dist_arg, sal_prop)):
        if dist[idx, num] < DISTHRES:
            template[sal.coords[:, 0], sal.coords[:, 1]] = nuc_prop[idx].label ### IS IT FINDING EVERYTHING IDENTIFIED?? 
    comb = np.max(np.dstack((template, nuc)), axis=2).astype(np.uint16)
    return template, comb, nuc_prop, nuc_loc

def label_nearest_sep_bacs(img, label, nuc, DISTHRES=25):
    """Label objects to the nearest nuc.
    """
    nuc_prop = regionprops(nuc, img, cache=False) #extract all region props from nuc seg
    sal_prop = regionprops(label, img, cache=False) #extract all region props from bacteria sec
    nuc_loc = [i.centroid for i in regionprops(nuc, img, cache=False)] # pull centroid for each nucleus
    sal_loc = [i.centroid for i in regionprops(label, img, cache=False)] # pull centrodi for each bac label 
    dist = pairwise_distance(nuc_loc, sal_loc) # calculate distance between nucleus and e coli 

    min_dist_arg = np.argmin(dist, axis=0) 

    template = np.zeros(img.shape, np.uint16)

    for num, (idx, sal) in enumerate(zip(min_dist_arg, sal_prop)):
        
        if dist[idx, num] < DISTHRES:
            print num
            print idx
            print sal
            template[sal.coords[:, 0], sal.coords[:, 1]] = nuc_prop[idx].label ### IS IT FINDING EVERYTHING IDENTIFIED?? 
    comb = np.max(np.dstack((template, nuc)), axis=2).astype(np.uint16)
    
    template_2 = skim_label(template)
    template_3 = np.zeros(img.shape,np.uint16)
    template_props = regionprops(template_2,img,cache=False)
    for temp in template_props:
        template_3[temp.coords[:,0],temp.coords[:,1]] = str(template[temp.coords[0,0],temp.coords[0,1]])+str(template_2[temp.coords[0,0],temp.coords[0,1]])
    
    return template, template_3, comb, nuc_prop, nuc_loc

def label_nearest_fixed (img, label, nuc, DISTHRES=25):
    """Label objects to the nearest nuc.
    """
    nuc_prop = regionprops(nuc, img, cache=False) #extract all region props from nuc seg
    sal_prop = regionprops(label, img, cache=False) #extract all region props from bacteria sec
    
    nuc_loc = [i.centroid for i in regionprops(nuc, img, cache=False)] # pull centroid for each nucleus
    sal_loc = [i.centroid for i in regionprops(label, img, cache=False)] # pull centrodi for each bac label 
    
    dist = pairwise_distance(nuc_loc, sal_loc) # calculate distance between nucleus and e coli 
    min_dist_arg = np.argmin(dist, axis=0) 
    template = np.zeros(img.shape, np.uint16)
    
    for idx, sal in zip(min_dist_arg, sal_prop):
        if 1:
            template[sal.coords[:, 0], sal.coords[:, 1]] = nuc_prop[idx].label
    comb = np.max(np.dstack((template, nuc)), axis=2).astype(np.uint16)
    return template, comb, nuc_prop, nuc_loc

def judge_bad(csig, psig, THRESCHANGE):
    ''' From Covertrack. Trying to adapt this to CellTK. 08/17/18.
    '''
    if csig - psig > THRESCHANGE:
        return True
    else:
        return False


def repair_sal(img, pimg, comb, pcomb, label, nuc_prop, nuc_loc, THRESCHANGE=1000):
    ''' From Covertrack. Trying to adapt this to CellTK. 08/17/18.
    '''

    # repair
    prev = regionprops(pcomb, pimg, cache=False) # previous image props
    prev_label = [i.label for i in prev] # previous image labels
    
    curr = regionprops(comb, img, cache=False) # current image 
    curr_label = [i.label for i in curr] # current image labels 

    store = [] 

    for cell in curr: # iterate over every cell in the current image 

        curr_sig = cell.mean_intensity * cell.area # the way we define a bacteria is the cell's mean intensity x area 
        
        if cell.label in prev_label: # if we recognize this bacteria already
            p_cell = prev[prev_label.index(cell.label)]  # find it in the last image 
            prev_sig = p_cell.mean_intensity * p_cell.area # store it's cell info 
        else: # if it's a new cell, we don't care about it 
            break
        
        if np.any(label == cell.label): # If we indeed found a cell across two frames, calculate it's difference in cell properties
            store.append(curr_sig - prev_sig) #store it's change from last frame to this frame in our areaxmean metric 
        
        if judge_bad(curr_sig, prev_sig, THRESCHANGE):  # If it's changed more than our desired threshold [# or diff ratio?] try to assign it to a neighbor
            print "I JUDGE BAD!"
            print cell.label 
            for rp in regionprops(skilabel(label == cell.label), img, cache=False): # for each property in regions props 
                dist = pairwise_distance((rp.centroid,), nuc_loc)[0] # calculate how far bac is from nucleus 
                for num in range(1, 4): # for 1,2,3,4 
                    
                    neighbor_nuc = nuc_prop[np.argsort(dist)[num]] #find the neighbor cell 
                    neiid = neighbor_nuc.label
                    nei_curr = curr[curr_label.index(neiid)] # find the current neighbor cell 
                    
                    if neiid not in prev_label: # don't bother if can't find neighbor 
                        break

                    nei_prev = prev[prev_label.index(neiid)] # find where the enighbor was in the last frame
                    nei_curr_sig = nei_curr.mean_intensity * nei_curr.area 
                    nei_prev_sig = nei_prev.mean_intensity * nei_prev.area
                    
                    if judge_bad(nei_curr_sig, nei_prev_sig, THRESCHANGE): # if the neighbor also changed by more than the treshold
                        label[rp.coords[:, 0], rp.coords[:, 1]] = neiid # swap out the neighbors 
                        print "I relabeled something"
                        print cell.label
                        break
                    else:
                        pass
    return label
