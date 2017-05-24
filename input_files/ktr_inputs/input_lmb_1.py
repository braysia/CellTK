
"""
output_folder

"""

from os.path import join, dirname, abspath


OUTPUT_DIR = '/home/output/LMB/Pos005'

op000 = dict(function='flatfield_references', inputdir='/home/KTRimages/LMB/Pos005/*DAPI*', ff_paths="/home/KTRimages/LMB/FF/*DAPI*", exp_corr=True)
op001 = dict(function='flatfield_references', inputdir='/home/KTRimages/LMB/Pos005/*YFP*', ff_paths="/home/KTRimages/LMB/FF/*YFP*", exp_corr=True)
op002 = dict(function='align', CROP=0.15, inputdir=["op000/*DAPI*", "op001/*YFP*"])

op003 = [dict(function='histogram_match', inputdir='op002/*DAPI*'), 
         dict(function="curvature_anisotropic_smooth", NITER=30, output_folder='DAPI')]

op004 = dict(function='gaussian_laplace', NEG=True, SIGMA=2.5)
op005 = dict(function='adaptive_thres', FIL1=20, R1=300)

op006 = dict(function='propagate_multisnakes', NITER=20, lambda2=3, inputdir='DAPI')

op007 = [dict(function='run_lap', MASSTHRES=0.25, DISPLACEMENT=20, inputdir='DAPI', labels_folder='op005'),
         dict(function='nearest_neighbor', DISPLACEMENT=20, MASSTHRES=0.25),
         dict(function='track_neck_cut', MASSTHRES=0.25),
         dict(function='track_neck_cut', THRES_ANGLE=160, DISPLACEMENT=20),
         dict(function='nearest_neighbor', DISPLACEMENT=60, MASSTHRES=0.1)]
op008 = [dict(function='gap_closing'),
         dict(function='cut_short_traces', minframe=100, output_folder='nuc')]

op009 = dict(function='ring_dilation_above_offset_buffer', RINGWIDTH=2, OFFSET=200, inputdir='op002/*YFP*', output_folder='cyto')

op010 = dict(function='apply', ch_folders=['DAPI', 'op002/*YFP*'], obj_folders=['nuc', 'cyto'], ch_names=['DAPI', 'YFP'])