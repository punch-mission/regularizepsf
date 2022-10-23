import numpy as np

from psfpy import simple_psf


@simple_psf
def constrained_gaussian(x, y, x0, y0, sigma_x, sigma_y):
    return np.exp(-(np.square(x-x0)/(2*np.square(sigma_x)) + np.square(y-y0)/(2*np.square(sigma_y))))


@simple_psf
def elliptical_gaussian(x, y, height, x0, y0, a, b, c):
    return height*np.exp(-(a*np.square(x-x0) + 2*b*(x-x0)*(y-y0) + c*np.square(y-y0)))
