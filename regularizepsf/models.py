import numpy as np

from regularizepsf import simple_psf


@simple_psf
def constrained_gaussian(x, y, x0=0, y0=0, sigma_x=1, sigma_y=1):
    return np.exp(-(np.square(x-x0)/(2*np.square(sigma_x)) + np.square(y-y0)/(2*np.square(sigma_y))))


@simple_psf
def elliptical_gaussian(x, y, height=1, x0=0, y0=0, a=1, b=0, c=1):
    return height*np.exp(-(a*np.square(x-x0) + 2*b*(x-x0)*(y-y0) + c*np.square(y-y0)))
