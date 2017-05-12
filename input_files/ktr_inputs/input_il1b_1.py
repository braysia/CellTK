
"""
output_folder

"""

from os.path import join, dirname, abspath


OUTPUT_DIR = 'output/IL1B/Pos006'

op0000 = dict(function='flatfield_references', inputdir='data/KTRimages/IL1B/Pos006/*DAPI*', ff_paths="data/KTRimages/IL1B/FF/*DAPI*", output_folder='op000')
op0001 = dict(function='flatfield_references', inputdir='data/KTRimages/IL1B/Pos006/*YFP*', ff_paths="data/KTRimages/IL1B/FF/*YFP*", output_folder='op000')
op0002 = dict(function='flatfield_references', inputdir='data/KTRimages/IL1B/Pos006/*TRITC*', ff_paths="data/KTRimages/IL1B/FF/*TRITC*", output_folder='op000')

op001 = dict(function='align', CROP=0.15, inputdir=["op000/*DAPI*", "op000/*YFP*", "op000/*TRITC*"])
op002 = [dict(function='histogram_match', inputdir='op001/*DAPI*'), 
         dict(function="curvature_anisotropic_smooth", NITER=30, output_folder='DAPI')]

op003 = dict(function='gaussian_laplace', NEG=True, SIGMA=2.5)
op004 = dict(function='adaptive_thres', FIL1=14, R1=300)

op005 = dict(function='propagate_multisnakes', NITER=20, lambda2=10, inputdir=join(OUTPUT_DIR, 'DAPI'))

op006 = [dict(function='run_lap', MASSTHRES=0.25, DISPLACEMENT=25), 
         dict(function='track_neck_cut', MASSTHRES=0.25),
         dict(function='track_neck_cut', THRES_ANGLE=160, DISPLACEMENT=20),
         dict(function='nearest_neighbor', DISPLACEMENT=60, MASSTHRES=0.1)]
op007 = [dict(function='gap_closing'),
         dict(function='cut_short_traces', minframe=100, output_folder='nuc')]

op009 = dict(function='ring_dilation_above_adaptive', inputdir='op001/*YFP*', output_folder='cyto')

op010 = dict(function='apply', ch_folders=['DAPI/*', 'op001/*YFP*', 'op001/*TRITC*'], obj_folders=['nuc', 'cyto'], ch_names=['DAPI', 'YFP', 'TRITC'])