from __future__ import annotations

import abc
import warnings
from typing import Any
from collections import namedtuple
from numbers import Real

import numpy as np
import deepdish as dd
from lmfit import Parameters, minimize, report_fit
from lmfit.minimizer import MinimizerResult
import sep
from photutils.detection import DAOStarFinder
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
from skimage.transform import resize, downscale_local_mean

from regularizepsf.psf import SimplePSF, VariedPSF, PointSpreadFunctionABC
from regularizepsf.exceptions import InvalidSizeError
from regularizepsf.corrector import calculate_covering


class PatchCollectionABC(metaclass=abc.ABCMeta):
    def __init__(self, patches: dict[Any, np.ndarray]):
        self._patches = patches
        if patches:
            shape = next(iter(patches.values())).shape
            # TODO: check that the patches are square
            self._size = shape[0]
        else:
            self._size = None

    def __len__(self):
        return len(self._patches)

    @classmethod
    @abc.abstractmethod
    def extract(cls, images: list[np.ndarray], coordinates: list, size: int) -> PatchCollectionABC:
        """Construct a PatchCollection from a set of images using the specified coordinates and patch size

        Parameters
        ----------
        images : list of np.ndarrays
            the images loaded
        coordinates : list
            A list of coordinates for the lower left pixel of each patch, specified in each type of PatchCollection
        size : int
            size of one side of the square patches extracted

        Returns
        -------
        np.ndarray
            the square patches extracted into a PatchCollection
        """

    def __getitem__(self, identifier: Any) -> np.ndarray:
        """Access a patch with square brackets

        Parameters
        ----------
        identifier : Any
            identifier for a given patch, specifically implemented for each PatchCollection

        Returns
        -------
        np.ndarray
            a patch's data
        """
        if identifier in self._patches:
            return self._patches[identifier]
        else:
            raise IndexError(f"{identifier} is not used to identify a patch in this collection.")

    def __contains__(self, identifier: Any) -> bool:
        """Determines if a patch is in the collection

        Parameters
        ----------
        identifier : Any
            identifier for a given patch, specifically implemented for each PatchCollection

        Returns
        -------
        bool
            True if patch with specified identifier is in the collection, False otherwise
        """
        return identifier in self._patches

    def add(self, identifier: Any, patch: np.ndarray) -> None:
        """Add a new patch to the collection

        Parameters
        ----------
        identifier : Any
            identifier for a given patch, specifically implemented for each PatchCollection

        patch : np.ndarray
            the data for a specific patch

        Returns
        -------
        None
        """
        if identifier in self._patches:
            # TODO: improve warning
            warnings.warn(f"{identifier} is being overwritten in this collection.", Warning)
        self._patches[identifier] = patch

        if self._size is None:
            self._size = patch.shape[0]
            # TODO : enforce square constraint

    @abc.abstractmethod
    def average(self, corners: np.ndarray, step: int, size: int, mode: str) -> PatchCollectionABC:
        """Construct a new PatchCollection where patches lying inside a new grid are averaged together

        Parameters
        ----------
        corners : np.ndarray
            an Nx2 `np.ndarray` of the lower left corners of the new patch grid
        step : int
            how far apart each corner patch is
        size : int
            dimension of the new (size, size) shaped square patches
        mode: str
            either average using "mean" or "median"

        Returns
        -------
        PatchCollectionABC
            a PatchCollection where data is sampled at the new grid
        """

    @abc.abstractmethod
    def fit(self, base_psf: SimplePSF, is_varied: bool = False) -> PointSpreadFunctionABC:
        """

        Parameters
        ----------
        base_psf
        is_varied

        Returns
        -------

        """

    def save(self, path: str) -> None:
        """Save the PatchCollection to a file

        Parameters
        ----------
        path : str
            where to save the patch collection

        Returns
        -------
        None
        """
        dd.io.save(path, self._patches)

    @classmethod
    def load(cls, path) -> PatchCollectionABC:
        """Load a PatchCollection from a file

        Parameters
        ----------
        path : str
            file path to load from

        Returns
        -------
        PatchCollectionABC
            the new patch collection
        """
        return cls(dd.io.load(path))

    def keys(self):
        """Gets identifiers for all patches"""
        return self._patches.keys()

    def values(self):
        """Gets values of all patches"""
        return self._patches.values()

    def items(self):
        """A dictionary like iterator over the patches"""
        return self._patches.items()

    def __next__(self):
        # TODO: implement
        pass

    def _fit_lmfit(self, base_psf: SimplePSF, initial_guesses: dict[str, Real]) -> dict[Any, MinimizerResult]:
        """Fit a patch using lmfit

        Parameters
        ----------
        base_psf : SimplePSF
            the PSF model to use in fitting
        initial_guesses : dict[str, Real]
            the initial guesses for all the PSF parameters

        Returns
        -------
        dict
            keys are the identifiers, values are the `MinimizerResults` from lmfit
        """
        initial = Parameters()
        for parameter in base_psf.parameters:
            initial.add(parameter, value=initial_guesses[parameter])

        xx, yy = np.meshgrid(np.arange(self._size), np.arange(self._size))

        results = dict()
        for identifier, patch in self._patches.items():
            results[identifier] = minimize(
                lambda current_parameters, x, y, data: data - base_psf(x, y, **current_parameters.valuesdict()),
                initial,
                args=(xx, yy, patch))
        return results


CoordinateIdentifier = namedtuple("CoordinateIdentifier", "image_index, x, y")


class CoordinatePatchCollection(PatchCollectionABC):
    """A representation of a PatchCollection that operates on pixel coordinates from a set of images

    """
    @classmethod
    def extract(cls, images: list[np.ndarray],
                coordinates: list[CoordinateIdentifier],
                size: int) -> PatchCollectionABC:
        out = cls(dict())

        # pad in case someone selects a region on the edge of the image
        padding_shape = ((size, size), (size, size))
        padded_images = [np.pad(image, padding_shape, mode='constant') for image in images]

        # TODO: prevent someone from selecting a region completing outside of the image
        for coordinate in coordinates:
            patch = padded_images[coordinate.image_index][coordinate.x+size:coordinate.x+2*size,
                                                          coordinate.y+size:coordinate.y+2*size]
            out.add(coordinate, patch)
        return out

    @classmethod
    def find_stars_and_average(cls, image_paths: list[str],
                               psf_size: int,
                               patch_size: int,
                               scale: int = 1,
                               average_mode: str = "median",
                               star_threshold: int = 3, hdu_choice=0):
        with fits.open(image_paths[0]) as hdul:
            image_shape = hdul[hdu_choice].data.shape

        this_collection = cls(dict())

        for i, image_path in enumerate(image_paths):
            with fits.open(image_path) as hdul:
                image = hdul[hdu_choice].data.astype(float)
            if image.shape != image_shape:
                raise ValueError(f"Images must all be the same shape. Found both {image_shape} and {image.shape}.")

            # if the image should be scaled then, do the scaling before anything else
            if scale != 1:
                interpolator = RectBivariateSpline(np.arange(image.shape[0]), np.arange(image.shape[1]), image)
                image = interpolator(np.linspace(0, image.shape[0], image.shape[0]*scale),
                                     np.linspace(0, image.shape[1], image.shape[1]*scale))

            background = sep.Background(image)
            image_background_removed = image - background
            image_star_coords = sep.extract(image_background_removed, star_threshold, err=background.globalrms)

            coordinates = [CoordinateIdentifier(i, int(y - psf_size * scale / 2), int(x - psf_size * scale / 2))
                           for x, y in zip(image_star_coords['x'], image_star_coords['y'])]

            # pad in case someone selects a region on the edge of the image
            padding_shape = ((psf_size * scale, psf_size * scale), (psf_size * scale, psf_size * scale))
            padded_image = np.pad(image, padding_shape, mode='constant', constant_values=np.median(image))

            for coordinate in coordinates:
                patch = padded_image[coordinate.x + scale * psf_size:coordinate.x + 2 * scale * psf_size,
                                     coordinate.y + scale * psf_size:coordinate.y + 2 * scale * psf_size]
                this_collection.add(coordinate, patch)

        corners = calculate_covering((image_shape[0]*scale, image_shape[1]*scale), patch_size*scale)
        averaged = this_collection.average(corners, patch_size*scale, psf_size*scale, mode=average_mode)

        if scale != 1:
            for coordinate, patch in averaged.items():
                averaged._patches[coordinate] = downscale_local_mean(averaged._patches[coordinate], (scale, scale))

            averaged._size = psf_size

        output = CoordinatePatchCollection(dict())
        for key, patch in averaged.items():
            output._patches[CoordinateIdentifier(key.image_index, key.x//scale, key.y//scale)] = patch

        return output

    def average(self, corners: np.ndarray, step: int, size: int,
                mode: str = "median") -> PatchCollectionABC:
        self._validate_average_mode(mode)
        pad_shape = self._calculate_pad_shape(size)

        if mode == "mean":
            mean_stack = {tuple(corner): np.zeros((size, size)) for corner in corners}
            counts = {tuple(corner): 0 for corner in corners}
        elif mode == "median":
            median_stack = {tuple(corner): [] for corner in corners}

        corners_x, corners_y = corners[:, 0], corners[:, 1]
        x_bounds = np.stack([corners_x, corners_x + step], axis=-1)
        y_bounds = np.stack([corners_y, corners_y + step], axis=-1)

        for identifier, patch in self._patches.items():
            # pad patch with zeros
            padded_patch = np.pad(patch / np.max(patch), pad_shape, mode='constant')

            # Determine which average region it belongs to
            center_x = identifier.x + self._size // 2
            center_y = identifier.y + self._size // 2
            x_matches = (x_bounds[:, 0] <= center_x) * (center_x < x_bounds[:, 1])
            y_matches = (y_bounds[:, 0] <= center_y) * (center_y < y_bounds[:, 1])
            match_indices = np.where(x_matches * y_matches)[0]

            # add to averages and increment count
            for match_index in match_indices:
                match_corner = tuple(corners[match_index])
                if mode == "mean":
                    mean_stack[match_corner] = np.nansum([mean_stack[match_corner], padded_patch], axis=0)
                    counts[match_corner] += 1
                elif mode == "median":
                    median_stack[match_corner].append(padded_patch)

        if mode == "mean":
            averages = {CoordinateIdentifier(None, corner[0], corner[1]): mean_stack[corner]/counts[corner]
                        for corner in mean_stack}
        elif mode == "median":
            averages = {CoordinateIdentifier(None, corner[0], corner[1]):
                            np.nanmedian(median_stack[corner], axis=0)
                            if len(median_stack[corner]) > 0 else np.zeros((size, size))
                        for corner in median_stack}
        return CoordinatePatchCollection(averages)

    def _validate_average_mode(self, mode: str):
        valid_modes = ['median', 'mean']
        if mode not in valid_modes:
            raise ValueError(f"Found a mode of {mode} but it must be in the list {valid_modes}.")

    def _calculate_pad_shape(self, size):
        pad_amount = size - self._size
        if pad_amount < 0:
            raise InvalidSizeError(f"The average window size (found {size})" 
                                   f"must be larger than the existing patch size (found {self._size}).")
        if pad_amount % 2 != 0:
            raise InvalidSizeError(f"The average window size (found {size})" 
                                   f"must be the same parity as the existing patch size (found {self._size}).")
        pad_shape = ((pad_amount//2, pad_amount//2), (pad_amount//2, pad_amount//2))
        return pad_shape

    def fit(self, base_psf: SimplePSF, is_varied: bool = False) -> PointSpreadFunctionABC:
        raise NotImplementedError("TODO")  # TODO: implement


