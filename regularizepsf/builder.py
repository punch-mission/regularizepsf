"""Functions for building PSF models from images."""

import pathlib
import multiprocessing
from collections.abc import Generator

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label
from skimage.transform import downscale_local_mean

from regularizepsf.exceptions import IncorrectShapeError, PSFBuilderError
from regularizepsf.image_processing import calculate_background, process_single_image
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
        elif len(images.shape) == 2:
            def generator() -> np.ndarray:
                while True:
                    yield images
            data_iterator = generator()
        else:
            msg = "Image data array must be 3D"
            raise IncorrectShapeError(msg)
    elif isinstance(images, list) and (isinstance(images[0], str) or isinstance(images[0], pathlib.Path)):
        def generator() -> np.ndarray:
            for image_path in images:
                yield image_path
        data_iterator = generator()
    else:
        msg = "Unsupported type for `images`"
        raise TypeError(msg)

    return data_iterator

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

    # if there were no patches at all, it will be filled with a np.nan instead of an array... we handle that carefully
    averages = {corner: patch if isinstance(patch, np.ndarray) else np.full((psf_size, psf_size), np.nan)
                for corner, patch in averages.items()}
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

    # we cannot allow any nans to propagate forward, so we fill them with 0 to indicate no response
    for corner, patch in averages.items():
        modified_patch = patch.copy()
        modified_patch[np.isnan(modified_patch)] = 0
        averages[corner] = modified_patch

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
              sep_mask: list[str] | list[pathlib.Path] | np.ndarray | Generator | None = None,
              hdu_choice: int | None = 0,
              num_workers: int | None = None,
              interpolation_scale: int = 1,
              star_threshold: int = 3,
              average_method: str = 'median',
              percentile: float = 50,
              saturation_threshold: float = np.inf,
              image_mask: np.ndarray | None = None,
              star_minimum: float = 0,
              star_maximum: float = np.inf,
              sqrt_compressed: bool = False,
              return_patches: bool = False) -> tuple[ArrayPSF, dict] | tuple[ArrayPSF, dict, dict]:
        """Build the PSF model.

        Parameters
        ----------
        images : list[str] | list[pathlib.Path] | np.ndarray | Generator
            Input images to use for PSF characterization
        sep_mask : list[str] | list[pathlib.Path] | np.ndarray | Generator | None
            Mask to use with source extraction (sep)
        hdu_choice : int | None
            HDU index to use when loading FITS input files
        num_workers : int | None
            Number of worker processes for multithreaded image processing, with None using all available CPUs
        interpolation_scale : int
            Interpolation scale to apply to input images after loading
        star_threshold : int
            Minimum threshold value for star detection using sep
        average_method : str
            Method for patch averaging (mean, percentile, or median)
        percentile : float
            Percentile value when specifying the percentile patch averaging method
        saturation_threshold : float
            Pixel value above which stars are considered saturated
        image_mask : np.ndarray | None
            Mask of pixels to ignore for PSF characterization in input images
        star_minimum : float
            Minimum threshold of center star for patch inclusion, in units of input data
        star_maximum : float
            Maximum threshold of center star for patch inclusion, in units of input data
        sqrt_compressed : bool
            Toggle to indicate if input data has been square-root compressed, and requires decompression
        return_patches : bool
            Toggle to return computed patches alongside model output

        Returns
        -------
        (ArrayPSF, dict)
            Array PSF and the counts of stars in each component

        """
        data_iterator = _convert_to_generator(images, hdu_choice=hdu_choice)

        if sep_mask is None:
            def generator() -> None:
                while True:
                    yield None
            mask_iterator = generator()
        else:
            mask_iterator = _convert_to_generator(sep_mask, hdu_choice=hdu_choice)

        args = [
            (i, image, star_mask, interpolation_scale, self.psf_size,
             star_threshold, saturation_threshold, image_mask, hdu_choice,
             star_minimum, star_maximum, sqrt_compressed)
            for i, (image, star_mask) in enumerate(zip(data_iterator, mask_iterator))
        ]

        with multiprocessing.Pool(processes = num_workers) as pool:
            results = pool.map(process_single_image, args)

        patches = {}
        image_shape = None
        for patch, data_shape in results:
            if image_shape is None:
                image_shape = data_shape
            elif image_shape != data_shape:
                msg = ("Images must all be the same shape."
                       f"Found both {image_shape} and {data_shape}.")
                raise PSFBuilderError(msg)

            patches.update(patch)

        corners = calculate_covering((image_shape[0] * interpolation_scale,
                                      image_shape[1] * interpolation_scale),
                                     self.psf_size * interpolation_scale)
        averaged_patches, counts = _average_patches(patches, corners,
                                                    method=average_method, percentile=percentile)

        values_coords = []
        values_array = np.zeros((len(averaged_patches), self.psf_size, self.psf_size))
        for i, (coordinate, patch) in enumerate(averaged_patches.items()):
            if interpolation_scale != 1:
                patch = downscale_local_mean(patch,(interpolation_scale, interpolation_scale))
            values_coords.append(coordinate)

            patch_background = calculate_background(patch)
            patch -= patch_background

            patch[patch == 0] = np.nan

            patch_central_value = patch[patch.shape[0]//2, patch.shape[1]//2]
            this_value_mask = patch < (0.005 * patch_central_value)
            this_value_mask = binary_erosion(this_value_mask, border_value = 1)

            patch[this_value_mask] = np.nan

            patch_zeroed = np.copy(patch)
            patch_zeroed[~np.isfinite(patch_zeroed)] = 0

            patch_labeled = label(patch_zeroed)[0]
            psf_core_mask = patch_labeled == patch_labeled[patch_labeled.shape[0]//2,patch_labeled.shape[1]//2]

            psf_core_mask = binary_dilation(psf_core_mask)

            patch_corrected = patch_zeroed * psf_core_mask
            patch_corrected = patch_corrected / np.nansum(patch_corrected)

            values_array[i,:,:] = patch_corrected

        if return_patches:
            return ArrayPSF(IndexedCube(values_coords, values_array)), counts, patches
        else:
            return ArrayPSF(IndexedCube(values_coords, values_array)), counts
