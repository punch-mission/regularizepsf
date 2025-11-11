import numpy as np
import scipy
import sep
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation, binary_erosion


def calculate_background(patch: np.ndarray) -> np.ndarray:
    patch_y, patch_x = np.indices(patch.shape)

    mask = patch != 0
    mask = binary_erosion(mask)
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    mask = binary_dilation(mask) & ~mask

    value_center = patch[patch.shape[1]//2, patch.shape[0]//2]
    mask = mask & (patch < value_center)

    A = np.c_[patch_x[mask], patch_y[mask], np.ones_like(patch_x[mask])]
    coefficients, _, _, _ = scipy.linalg.lstsq(A, patch[mask])
    background = coefficients[0] * patch_x + coefficients[1] * patch_y + coefficients[2]
    background[patch == 0] = np.nan

    return background


def _scale_image(image_path, interpolation_scale, hdu_choice):
    with fits.open(image_path) as hdul:
        image = hdul[hdu_choice].data.astype(float)
    interpolator = RectBivariateSpline(np.arange(image.shape[0]),
                                       np.arange(image.shape[1]),
                                       image)
    image = interpolator(np.linspace(0,
                                     image.shape[0] - 1,
                                     1 + (image.shape[0] - 1) * interpolation_scale),
                         np.linspace(0,
                                     image.shape[1] - 1,
                                     1 + (image.shape[1] - 1) * interpolation_scale))
    return image


def _find_patches(image, star_threshold, star_mask, interpolation_scale, psf_size, i,
                  saturation_threshold: float = np.inf, image_mask: np.ndarray | None = None,
                  star_minimum: float = 0, star_maximum: float = np.inf):
    background = sep.Background(image)
    image_background_removed = image - background

    try:
        image_star_coords = sep.extract(image_background_removed,
                                        star_threshold,
                                        err=background.globalrms,
                                        mask=star_mask)
    except:
        return {"x":[], "y":[]}

    coordinates = [(i,
                    int(round(x - psf_size * interpolation_scale / 2)),
                    int(round(y - psf_size * interpolation_scale / 2)))
                   for x, y in zip(image_star_coords["y"], image_star_coords["x"], strict=True)]

    # pad in case someone selects a region on the edge of the image
    padding_shape = ((psf_size * interpolation_scale, psf_size * interpolation_scale),
                     (psf_size * interpolation_scale, psf_size * interpolation_scale))
    padded_image = np.pad(image,
                          padding_shape,
                          mode="reflect")

    # the mask indicates which pixel should be ignored in the calculation
    if image_mask is not None:
        padded_mask = np.pad(image_mask, padding_shape, mode='reflect')
    else:  # if no mask is provided, we create an empty mask
        padded_mask = np.zeros_like(padded_image, dtype=bool)

    patches = {}
    for coordinate in coordinates:
        patch = padded_image[coordinate[1] + interpolation_scale * psf_size:
                             coordinate[1] + 2 * interpolation_scale * psf_size,
                coordinate[2] + interpolation_scale * psf_size:
                coordinate[2] + 2 * interpolation_scale * psf_size]
        mask_patch = padded_mask[coordinate[1] + interpolation_scale * psf_size:
                             coordinate[1] + 2 * interpolation_scale * psf_size,
                            coordinate[2] + interpolation_scale * psf_size:
                            coordinate[2] + 2 * interpolation_scale * psf_size]
        
        # Separately background subtract each patch
        background_patch = calculate_background(patch)
        patch_background_subtracted = patch - background_patch
        patch_background_subtracted[patch == 0] = np.nan

        # we do not add patches that have saturated pixels
        if np.all(patch_background_subtracted < saturation_threshold):
            patch_background_subtracted[mask_patch] = np.nan
            patches[coordinate] = patch_background_subtracted

        # # we do not add patches that have central stars outside of our defined limits
        center = (patch_background_subtracted.shape[1] // 2, patch_background_subtracted.shape[0] // 2)
        if (patch_background_subtracted[center] < star_minimum) | (patch_background_subtracted[center] > star_maximum):
            patch_background_subtracted[mask_patch] = np.nan
            patches[coordinate] = patch_background_subtracted
    return patches


def process_single_image(args):
    """Process a single image to extract patches.
    
    Parameters
    ----------
    args : tuple
        Tuple containing (i, image, star_mask, interpolation_scale, psf_size, 
        star_threshold, saturation_threshold, image_mask)
    
    Returns
    -------
    dict
        Dictionary of patches found in the image
    """
    i, image_path, star_mask, interpolation_scale, psf_size, star_threshold, saturation_threshold, image_mask, hdu_choice, star_minimum, star_maximum, sqrt_compressed = args
    
    if interpolation_scale != 1:
        image = _scale_image(image_path, interpolation_scale=interpolation_scale, hdu_choice=hdu_choice)
    else:
        # Load the image directly when no scaling is needed
        with fits.open(image_path) as hdul:
            header = hdul[hdu_choice].header
            scale = header['SCALE']
            image = ((hdul[hdu_choice].data.astype(float))**2)/scale

    return _find_patches(image, star_threshold, star_mask, interpolation_scale, psf_size, i,
                        saturation_threshold, image_mask)
