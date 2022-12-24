# Quickstart

## Installing
Open a terminal and run `pip install regularizepsf`. 

## Overview of the technique
TODO: fill in the basic theory

## Demo: extracting a simple PSF model
This tutorial uses an image from the Demonstration Airglow-Subtracted Heliospheric
imager (DASH) as an example. For details on DASH see Section 4.2 of the [accompanying paper](`https://ui.adsabs.harvard.edu/abs/2022arXiv221202594H`).
Here is a visualization of the DASH data:

TODO: put in figure of DAsH

### Defining input parameters
Before you can correct an image, you need a PSF model for the system that collected the image. 
You can either define a **functional model** or an **array model.** A functional
model is defined by an equation whereas an array model uses the data directly as the model. 
For most purposes, we recommend an array model because deriving the correct functional form 
can be tricky. Thus, this quickstart tutorial walks through the array model PSF model form.

First, we need to define both the `psf_size` and the `patch_size`. This technique breaks the image into 
overlapping square neighborhoods of length `patch_size` on each side. Selecting the `patch_size` is discussed
later (TODO: where?). In this case a `patch_size` of 256 indicates we use a `256x255` neighborhood size. However, we know for 
this instrument that the PSF is more compact than this, i.e. a star will not contribute light over the full neighborhood. 
For computational efficiency reasons, we thus use a `psf_size` of `32x32`. For each star, a box centered on the star
of size `32x32` pixels is extracted. These are averaged to create the PSF model. The region outside the `32x32` box but 
within `256x256` pixel window is filled with the median of the image.

Finally, we need to set the target PSF model's parameters. In this case, we will simply use a 
symmetric Gaussian and thus have one parameter: the full-width-half-maximum (FWHM). Selection of the FWHM for the target
model is discussed elsewhere in the documentation (TODO: where?). We set this at 3.25 pixels. 
```py
import numpy as np
from astropy.io import fits
from regularizepsf import CoordinatePatchCollection, simple_psf

# Define the parameters and image to use
psf_size = 32
patch_size = 256
target_fwhm = 3.25
image_fn = "path/to/image.fits"
```

### Building the target PSF
As discussed above, we need a target PSF model to establish uniformity in the output PSF. First, we define
a Python function that has inputs of `x` and `y` and outputs the value of the PSF at `(x,y)`. 
This is a functional model and is thus decorated with the `simple_psf` decorator. We evaluate the model over the full 
patch size. 
```py
# Define the target PSF
@simple_psf
def target(x, y, 
           x0=patch_size / 2, y0=patch_size / 2, 
           sigma_x= target_fwhm / 2.355, sigma_y= target_fwhm / 2.355):
    return np.exp(-(np.square(x - x0) / (2 * np.square(sigma_x)) 
                    + np.square(y - y0) / (2 * np.square(sigma_y))))

target_evaluation = target(*np.meshgrid(np.arange(patch_size), np.arange(patch_size)))
```

### Building the ArrayCorrector
Everything else is handled in the `find_stars_and_average` method on the `CoordinatePathCollection`. 
It takes a list of images, the `psf_size`, and the `patch_size`. The function's signature is:

```{eval-rst}
.. autofunction:: regularizepsf.fitter.CoordinatePatchCollection.find_stars_and_average
```


We convert from a `CoordinatePatchCollection` object
to an `ArrayCorrector` using the `to_array_corrector` method. It requires an evaluated target PSF, which we constructed
in the previous section.
```py

# Extract all the stars from that image and create a PSF model with a target PSF
array_corrector =  CoordinatePatchCollection.find_stars_and_average([image_fn], 
                                                                     psf_size, 
                                                                     patch_size).to_array_corrector(target_evaluation)
```

The signature of `to_array_corrector` is:
```{eval-rst}
.. autofunction:: regularizepsf.fitter.CoordinatePatchCollection.to_array_corrector
```

### Performing the correction
Now we have all the components. It's time to correct the image! We open it as a `numpy` array and then use the
`array_corrector` to correct. 

```py
# Load the image for correcting
with fits.open(image_fn) as hdul:
    data = hdul[0].data
    
# See the corrected result! 
corrected = array_corrector.correct_image(data, alpha=2.0, epsilon=0.3)
```

TODO: put the correction comparison images in

The signature of `correct_image` is:
```{eval-rst}
.. autofunction:: regularizepsf.corrector.ArrayCorrector.correct_image
```

You can save the `ArrayCorrector` for future usage instead of having to derive it for each image. 