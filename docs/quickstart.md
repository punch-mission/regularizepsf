# Quickstart

## Installing
Installation should be as simple as `pip install regularizepsf`. 

## Extracting a simple PSF model
Before you can correct an image, you need a PSF model for it. 
You can either define a functional model or an array model. A functional
model is defined by an equation whereas an array model uses the data directly as the model. 
For most purposes, we recommend an array model because deriving the correct functional form 
can be tricky.


## Correcting an image with a PSF ArrayCorrector

```py
from regularizepsf.fitter import CoordinatePatchCollection
patch_size, psf_size = 256, 32
image_fn = "path/to/image.fits"

averaged =  CoordinatePatchCollection.find_stars_and_average([image_fn], 
                                                             psf_size, 
                                                             patch_size)

@simple_psf
def target(x, y, x0=patch_size / 2, y0=patch_size / 2, sigma_x=3.25 / 2.355, sigma_y=3.25 / 2.355):
    return np.exp(-(np.square(x - x0) / (2 * np.square(sigma_x)) + np.square(y - y0) / (2 * np.square(sigma_y))))

evaluation_dictionary = {(identifier.x, identifer.y): patch for identifier, patch in averaged.items()}
target_evaluation = target(*np.meshgrid(np.arange(patch_size), np.arange(patch_size)))
array_corrector = ArrayCorrector(evaluation_dictionary, target_evaluation)

with fits.open(image_fn) as hdul:
    data = hdul[0].data
    
corrected = array_corrector.correct_image(data, alpha=2.0, epsilon=0.3)
```