from os.path import join, dirname, abspath

OUTPUT_DIR = '/home/output/AnisoInh/Pos0'

op000 = [dict(function='flatfield_references', inputdir="/home/KTRimages/AnisoInh/Pos0/img*DAPI*", ff_paths="/home/KTRimages/AnisoInh/FF/img*DAPI*"), 
         dict(function='histogram_match', output_folder='DAPI')]
op001 = dict(function="curvature_anisotropic_smooth", NITER=30)

op002 = dict(function='gaussian_laplace', NEG=True, SIGMA=2.5)
op003 = dict(function='adaptive_thres', FIL1=14, R1=300)

op004 = dict(function='propagate_multisnakes', NITER=20, lambda2=10, inputdir=join(OUTPUT_DIR, 'DAPI'))

op005 = [dict(function='run_lap', MASSTHRES=0.25, DISPLACEMENT=25), 
         dict(function='track_neck_cut', MASSTHRES=0.25),
         dict(function='track_neck_cut', THRES_ANGLE=160, DISPLACEMENT=20),
         dict(function='nearest_neighbor', DISPLACEMENT=60, MASSTHRES=0.15)]
op006 = [dict(function='gap_closing'),
         dict(function='cut_short_traces', minframe=100, output_folder='nuc')]

op007 = dict(function='flatfield_references', inputdir='/home/KTRimages/AnisoInh/Pos0/img*YFP*',
             ff_paths='/home/KTRimages/AnisoInh/FF/img*YFP*', output_folder='YFP')
op008 = dict(function='ring_dilation_above_offset_buffer', OFFSET=200, inputdir='YFP', output_folder='cyto')

op009 = dict(function='apply', ch_folders=['DAPI', 'YFP'], obj_folders=['nuc', 'cyto'])