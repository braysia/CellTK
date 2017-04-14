from utils.filters import calc_lapgauss


def gaussian_laplace(img, SIGMA=2.5):
    return calc_lapgauss(img, SIGMA)