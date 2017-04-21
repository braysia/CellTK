import numpy as np
from scipy.ndimage import imread as imread0


def imread(path):
    if isinstance(path, tuple) or isinstance(path, list):
        st = []
        for p in path:
            st.append(imread0(p))
        return np.dstack(st)
    else:
        return imread0(path)