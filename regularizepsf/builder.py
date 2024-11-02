"""Functions for building PSF models from images."""

import pathlib
from collections.abc import Generator

import numpy as np
import sep_pjw as sep
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
from skimage.transform import downscale_local_mean

from regularizepsf.exceptions import IncorrectShapeError, PSFBuilderError
from regularizepsf.psf import ArrayPSF
from regularizepsf.util import IndexedCube, calculate_covering


def _convert_to_generator(images:  list[pathlib.Path] | np.ndarray | Generator,
                          hdu_choice: int | None = None) -> Generator:
    if isinstance(images, Generator):
        data_iterator = images
    elif isinstance(images, np.ndarray):
        if len(images.shape) == 3:
            def generator() -> np.ndarray:
                yield from images
            data_iterator = generator()
        else:
            msg = "Image data array must be 3D"
            raise IncorrectShapeError(msg)
    elif isinstance(images, list) and (isinstance(images[0], str) or isinstance(images[0], pathlib.Path)):
        def generator() -> np.ndarray:
            for image_path in images:
                with fits.open(image_path) as hdul:
                    yield hdul[hdu_choice].data.astype(float)
        data_iterator = generator()
    else:
        msg = "Unsupported type for `images`"
        raise TypeError(msg)

    return data_iterator

def _scale_image(image, interpolation_scale):
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

def _find_patches(image, star_threshold, star_mask, interpolation_scale, psf_size, i):
    background = sep.Background(image)
    image_background_removed = image - background
    image_star_coords = sep.extract(image_background_removed,
                                    star_threshold,
                                    err=background.globalrms,
                                    mask=star_mask)

    coordinates = [(i,
                    int(round(x - psf_size * interpolation_scale / 2)),
                    int(round(y - psf_size * interpolation_scale / 2)))
                   for x, y in zip(image_star_coords["y"], image_star_coords["x"], strict=True)]

    # pad in case someone selects a region on the edge of the image
    padding_shape = ((psf_size * interpolation_scale, psf_size * interpolation_scale),
                     (psf_size * interpolation_scale, psf_size * interpolation_scale))
    padded_image = np.pad(image_background_removed,
                          padding_shape,
                          mode="reflect")

    patches = {}
    for coordinate in coordinates:
        patch = padded_image[coordinate[1] + interpolation_scale * psf_size:
                             coordinate[1] + 2 * interpolation_scale * psf_size,
                coordinate[2] + interpolation_scale * psf_size:
                coordinate[2] + 2 * interpolation_scale * psf_size]
        patches[coordinate] = patch

    return patches

def _find_matches(coordinate, x_bounds, y_bounds, psf_size):
    center_x = coordinate[1] + psf_size // 2
    center_y = coordinate[2] + psf_size // 2
    x_matches = (x_bounds[:, 0] <= center_x) * (center_x < x_bounds[:, 1])
    y_matches = (y_bounds[:, 0] <= center_y) * (center_y < y_bounds[:, 1])
    match_indices = np.where(x_matches * y_matches)[0]
    return match_indices

def _average_patches_by_mean(patches, corners, x_bounds, y_bounds, psf_size):
    accumulator = {tuple(corner): np.zeros((psf_size, psf_size))
                        for corner in corners}
    accumulator_counts = {tuple(corner): np.zeros((psf_size, psf_size))
                       for corner in corners}
    counts = {tuple(corner): 0 for corner in corners}

    for coordinate, patch in patches.items():
        patch = patch / patch[psf_size // 2, psf_size // 2]  # normalize so the star brightness is always 1
        match_indices = _find_matches(coordinate, x_bounds, y_bounds, psf_size)

        for match_index in match_indices:
            match_corner = tuple(corners[match_index])
            accumulator[match_corner] = np.nansum([accumulator[match_corner], patch], axis=0)
            accumulator_counts[match_corner] += np.isfinite(patch)
            counts[match_corner] += 1

    averages = {(corner[0], corner[1]):
                    accumulator[corner] / accumulator_counts[corner]
                for corner in accumulator}

    return averages, counts

def _average_patches_by_percentile(patches, corners, x_bounds, y_bounds, psf_size, percentile: float=50):
    if percentile == 50:
        percentile_method = lambda d: np.nanmedian(d, axis=0)
    else:
        percentile_method = lambda d: np.nanpercentile(d, percentile, axis=0)

    stack = {tuple(corner): [] for corner in corners}
    counts = {tuple(corner): 0 for corner in corners}

    for coordinate, patch in patches.items():
        patch = patch / patch[psf_size // 2, psf_size // 2]  # normalize so the star brightness is always 1
        match_indices = _find_matches(coordinate, x_bounds, y_bounds, psf_size)

        for match_index in match_indices:
            match_corner = tuple(corners[match_index])
            stack[match_corner].append(patch)
            counts[match_corner] += 1

    averages = {(corner[0], corner[1]): percentile_method(stack[corner]) for corner in stack}
    return averages, counts

def _average_patches(patches, corners, method='mean', percentile: float=None):
    psf_size = next(iter(patches.values())).shape[0]
    corners_x, corners_y = corners[:, 0], corners[:, 1]
    x_bounds = np.stack([corners_x, corners_x + psf_size], axis=-1)
    y_bounds = np.stack([corners_y, corners_y + psf_size], axis=-1)

    if method == 'mean':
        averages, counts = _average_patches_by_mean(patches, corners, x_bounds, y_bounds, psf_size)
    elif method == 'percentile':
        averages, counts = _average_patches_by_percentile(patches, corners, x_bounds, y_bounds, psf_size, percentile)
    elif method == "median":
        averages, counts = _average_patches_by_percentile(patches, corners, x_bounds, y_bounds, psf_size, 50)
    else:
        raise PSFBuilderError(f"Unknown method {method}.")

    return averages, counts

class ArrayPSFBuilder:
    """A builder that will take a series of images and construct an ArrayPSF to represent their implicit PSF."""

    def __init__(self, psf_size: int) -> None:
        """Initialize an ArrayPSFBuilder."""
        self._psf_size = psf_size

    @property
    def psf_size(self):
        return self._psf_size

    def build(self,
              images: list[str] | list[pathlib.Path] | np.ndarray | Generator,
              star_masks: list[str] | list[pathlib.Path] | np.ndarray | Generator | None = None,
              hdu_choice: int | None = 0,
              interpolation_scale: int = 1,
              star_threshold: int = 3,
              average_method: str = 'median',
              percentile: float = 50) -> (ArrayPSF, dict):
        """Build the PSF model.

        Parameters
        ----------
        images :  list[pathlib.Path] | np.ndarray | Generator
            images to use

        Returns
        -------
        (ArrayPSF, dict)
            an array PSF and the counts of stars in each component

        """
        data_iterator = _convert_to_generator(images, hdu_choice=hdu_choice)

        if star_masks is None:
            def generator() -> None:
                while True:
                    yield None
            mask_iterator = generator()
        else:
            mask_iterator = _convert_to_generator(star_masks, hdu_choice=hdu_choice)

        # We'll store the first image's shape, and then make sure the others match.
        image_shape = None
        patches = {}
        for i, (image, star_mask) in enumerate(zip(data_iterator, mask_iterator, strict=False)):
            if image_shape is None:
                image_shape = image.shape
            elif image.shape != image_shape:
                msg = ("Images must all be the same shape."
                       f"Found both {image_shape} and {image.shape}.")
                raise PSFBuilderError(msg)

            # if the image should be scaled then, do the scaling before anything else
            if interpolation_scale != 1:
                image = _scale_image(image, interpolation_scale=1)

            # find stars using SEP
            patches.update(_find_patches(image, star_threshold, star_mask, interpolation_scale, self.psf_size, i))

        corners = calculate_covering((image_shape[0] * interpolation_scale,
                                      image_shape[1] * interpolation_scale),
                                     self.psf_size * interpolation_scale)
        averaged_patches, counts = _average_patches(patches, corners,
                                                    method=average_method, percentile=percentile)

        values_coords = []
        values_array = np.zeros((len(averaged_patches), self.psf_size, self.psf_size))
        for i, (coordinate, this_patch) in enumerate(averaged_patches.items()):
            if interpolation_scale != 1:
                this_patch = downscale_local_mean(this_patch,(interpolation_scale, interpolation_scale))
            values_coords.append(coordinate)
            values_array[i, :, :] = this_patch

        return ArrayPSF(IndexedCube(values_coords, values_array)), counts
