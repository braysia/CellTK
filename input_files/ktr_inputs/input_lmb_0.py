
"""
output_folder

"""

from os.path import join, dirname, abspath


OUTPUT_DIR = 'output/LMB/Pos004'

op000 = dict(function='flatfield_references', inputdir='data/KTRimages/LMB/Pos004/*DAPI*', ff_paths="data/KTRimages/LMB/FF/*DAPI*", exp_corr=True, output_folder='DAPI')
op001 = dict(function='flatfield_references', inputdir='data/KTRimages/LMB/Pos004/*YFP*', ff_paths="data/KTRimages/LMB/FF/*YFP*", output_folder='YFP')

op002 = [dict(function='histogram_match', inputdir='DAPI'), 
         dict(function="curvature_anisotropic_smooth", NITER=30, output_folder='DAPI')]

op003 = dict(function='gaussian_laplace', NEG=True, SIGMA=2.5)
op004 = dict(function='adaptive_thres', FIL1=14, R1=300)

op005 = dict(function='propagate_multisnakes', NITER=20, lambda2=10, inputdir='DAPI')

op006 = [dict(function='run_lap', MASSTHRES=0.25, DISPLACEMENT=20, inputdir='DAPI', labels_folder='op005'),
         dict(function='nearest_neighbor', DISPLACEMENT=20, MASSTHRES=0.25),
         dict(function='track_neck_cut', MASSTHRES=0.25),
         dict(function='track_neck_cut', THRES_ANGLE=160, DISPLACEMENT=20),
         dict(function='nearest_neighbor', DISPLACEMENT=60, MASSTHRES=0.1)]
op007 = [dict(function='gap_closing'),
         dict(function='cut_short_traces', minframe=100, output_folder='nuc')]

op009 = dict(function='ring_dilation_above_adaptive', inputdir='YFP', output_folder='cyto')

op010 = dict(function='apply', ch_folders=['DAPI', 'YFP'], obj_folders=['nuc', 'cyto'])