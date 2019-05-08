
# Remove if you don't need to save certain properties. 
PROP_SAVE = ['area', 'cell_id', 'convex_area', 'cv_intensity',
             'eccentricity', 'major_axis_length', 'minor_axis_length',
             'max_intensity', 'mean_intensity', 'median_intensity',
             'min_intensity', 'orientation', 'perimeter', 'solidity',
             'std_intensity', 'total_intensity', 'x',
             'y', 'parent', 'num_seg']

# If FRAME_REVTRACK=N, tracking will start in a reverse order
# from frame N, N-1...0 and then forward 1, 2, 3....
# This may help the adaptive segmentation cutting (e.g. track_neck_cut or watershed_distance)
FRAME_REVTRACK = 0

# segment.py runs a moderate clean up at the end.
RUN_CLEAN = True

# Parameters used in the clean up.
RADIUS, OPEN = [3, 100], 3

# The number of cells in a track should not exceed this value.
MAX_NUMCELL = 100000

# File name for saving used during apply.py
FILE_NAME = 'df'