import numpy as np
from astropy.io import fits
from regularizepsf import CoordinatePatchCollection, simple_psf

# Define the parameters and image to use
psf_size = 32
patch_size = 256
target_fwhm = 3.25
image_fn = "data/DASH.fits"

# Define the target PSF
@simple_psf
def target(x, y,
           x0=patch_size / 2, y0=patch_size / 2,
           sigma_x= target_fwhm / 2.355, sigma_y= target_fwhm / 2.355):
    return np.exp(-(np.square(x - x0) / (2 * np.square(sigma_x))
                    + np.square(y - y0) / (2 * np.square(sigma_y))))

target_evaluation = target(*np.meshgrid(np.arange(patch_size), np.arange(patch_size)))

# Extract all the stars from that image and create a PSF model with a target PSF
array_corrector = CoordinatePatchCollection.find_stars_and_average([image_fn],
                                                                   psf_size,
                                                                   patch_size).to_array_corrector(target_evaluation)

# Load the image for correcting
with fits.open(image_fn) as hdul:
    data = hdul[0].data.astype(float)

print("ready to correct")
# See the corrected result!
corrected = array_corrector.correct_image(data, alpha=2.0, epsilon=0.3)
