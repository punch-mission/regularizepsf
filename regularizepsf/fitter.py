from __future__ import annotations

import abc
import warnings
from collections import namedtuple
from collections.abc import Callable
from numbers import Real
from typing import Any, Dict, Generator, List, Optional, Tuple

import h5py
import numpy as np
import sep
from astropy.io import fits
from lmfit import Parameters, minimize
from lmfit.minimizer import MinimizerResult
from scipy.interpolate import RectBivariateSpline
from skimage.transform import downscale_local_mean

from regularizepsf.corrector import ArrayCorrector, calculate_covering
from regularizepsf.exceptions import InvalidSizeError
from regularizepsf.psf import PointSpreadFunctionABC, SimplePSF


class PatchCollectionABC(metaclass=abc.ABCMeta):
    def __init__(self, patches: dict[Any, np.ndarray], counts: Optional[dict[Any, int]] = None) -> None:
        self.patches = patches
        self.counts = counts
        if patches:
            shape = next(iter(patches.values())).shape
            # TODO: check that the patches are square
            self.size = shape[0]
        else:
            self.size = None

    def __len__(self) -> int:
        return len(self.patches)

    @classmethod
    @abc.abstractmethod
    def extract(cls, 
                images: list[np.ndarray], 
                coordinates: list, 
                size: int) -> PatchCollectionABC:
        """Construct a PatchCollection from a set of images 
        using the specified coordinates and patch size

        Parameters
        ----------
        images : list of np.ndarrays
            the images loaded
        coordinates : list
            A list of coordinates for the lower left pixel of each patch, 
                specified in each type of PatchCollection
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
            identifier for a given patch, specifically implemented 
            for each PatchCollection

        Returns
        -------
        np.ndarray
            a patch's data
        """
        if identifier in self.patches:
            return self.patches[identifier]
        else:
            msg = f"{identifier} is not used to identify a patch in this collection."
            raise IndexError(msg)

    def __contains__(self, identifier: Any) -> bool:
        """Determines if a patch is in the collection

        Parameters
        ----------
        identifier : Any
            identifier for a given patch, 
                specifically implemented for each PatchCollection

        Returns
        -------
        bool
            True if patch with specified identifier is in the collection, 
                False otherwise
        """
        return identifier in self.patches

    def add(self,
            identifier: Any,
            patch: np.ndarray,
            count: Optional[int] = None) -> None:
        """Add a new patch to the collection

        Parameters
        ----------
        identifier : Any
            identifier for a given patch, 
                specifically implemented for each PatchCollection
        patch : np.ndarray
            the data for a specific patch
        count : int
            Optionally, a corresponding item to add to the `counts` dictionary

        Returns
        -------
        None
        """
        if identifier in self.patches:
            # TODO: improve warning
            warnings.warn(f"{identifier} is being overwritten in this collection.",
                           Warning, stacklevel=2)
        self.patches[identifier] = patch

        if count is not None:
            self.counts[identifier] = count

        if self.size is None:
            self.size = patch.shape[0]
            # TODO : enforce square constraint

    @abc.abstractmethod
    def average(self, 
                corners: np.ndarray, 
                step: int, 
                size: int,
                mode: str) -> PatchCollectionABC:
        """Construct a new PatchCollection where patches 
        lying inside a new grid are averaged together

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
    def fit(self, 
            base_psf: SimplePSF, 
            is_varied: bool = False) -> PointSpreadFunctionABC:
        """

        Parameters
        ----------
        base_psf
        is_varied

        Returns
        -------

        """

    @abc.abstractmethod
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


    @classmethod
    @abc.abstractmethod
    def load(cls, path: str) -> PatchCollectionABC:
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

    def keys(self) -> List:
        """Gets identifiers for all patches"""
        return self.patches.keys()

    def values(self) -> List:
        """Gets values of all patches"""
        return self.patches.values()

    def items(self) -> Dict:
        """A dictionary like iterator over the patches"""
        return self.patches.items()

    def _fit_lmfit(self, 
                   base_psf: SimplePSF, 
                   initial_guesses: dict[str, Real]) -> dict[Any, MinimizerResult]:
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

        xx, yy = np.meshgrid(np.arange(self.size), np.arange(self.size))

        results = {}
        for identifier, patch in self.patches.items():
            results[identifier] = minimize(
                lambda current_parameters, x, y, data:
                    data - base_psf(x, y, **current_parameters.valuesdict()),
                initial,
                args=(xx, yy, patch))
        return results


CoordinateIdentifier = namedtuple("CoordinateIdentifier", "image_index, x, y")


class CoordinatePatchCollection(PatchCollectionABC):
    """A representation of a PatchCollection that operates 
    on pixel coordinates from a set of images"""
    @classmethod
    def extract(cls, images: list[np.ndarray],
                coordinates: list[CoordinateIdentifier],
                size: int) -> PatchCollectionABC:
        out = cls({})

        # pad in case someone selects a region on the edge of the image
        padding_shape = ((size, size), (size, size))
        padded_images = [np.pad(image, padding_shape, mode="constant") 
                         for image in images]

        # TODO: prevent someone from selecting a region completing outside of the image
        for coordinate in coordinates:
            patch = padded_images[coordinate.image_index][coordinate.x+size:coordinate.x+2*size,
                                                          coordinate.y+size:coordinate.y+2*size]
            out.add(coordinate, patch)
        return out

    @classmethod
    def find_stars_and_average(cls, 
                               images: list[str] | np.ndarray | Generator,
                               psf_size: int,
                               patch_size: int,
                               interpolation_scale: int = 1,
                               average_mode: str = "median",
                               percentile: float = 10,
                               star_threshold: int = 3,
                               star_mask: Optional[list[str] | np.ndarray | Generator] = None,
                               hdu_choice: int=0) -> CoordinatePatchCollection:
        """Loads a series of images, finds stars in each, 
            and builds a CoordinatePatchCollection with averaged stars

        Parameters
        ----------
        images : List[str] or np.ndarray or Generator
            The images to be processed. Can be a list of FITS filenames, a
            numpy array of shape (n_images, ny, nx), or a Generator that yields
            each data array in turn.
        psf_size : int
            size of the PSF model to use
        patch_size : int
            square size that each PSF model applies to
        interpolation_scale : int
            if >1, the image are first scaled by this factor. 
                This results in stars being aligned at a subpixel scale
        average_mode : str
            "median", "percentile", or "mean": determines how patches are
            combined
        percentile : float
            If `average_mode` is `"percentile"`, use this percentile value
            (from 0 to 100)
        star_threshold : int
            SEP's threshold for finding stars. See `threshold`
                in https://sep.readthedocs.io/en/v1.1.x/api/sep.extract.html#sep-extract
        star_mask : List[str] or np.ndarray or Generator
            Masks to apply during star-finding. Can be a list of FITS filenames, a
            numpy array of shape (n_images, ny, nx), or a Generator that yields
            each mask array in turn. Where the mask pixel is `True`, the
            corresponding data array pixel will not be selected as a star. See
            `mask` in
            https://sep.readthedocs.io/en/v1.1.x/api/sep.extract.html#sep-extract
            for more details.
        hdu_choice : int
            Which HDU from each image will be used, 
                default of 0 is most common but could be 1 for compressed images

        Returns
        -------
        CoordinatePatchCollection
            An averaged star model built from the provided images

        Notes
        ------
        Using an `interpolation_scale` other than 1 
            for large images can dramatically slow down the execution.
        """
        if isinstance(images, Generator):
            data_iterator = images
        elif isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                def generator():
                    for image in images:
                        yield image
                data_iterator = generator()
            else:
                raise ValueError("Image data array must be 3D")
        elif isinstance(images, List) and isinstance(images[0], str):
            def generator():
                for image_path in images:
                    with fits.open(image_path) as hdul:
                        yield hdul[hdu_choice].data.astype(float)
            data_iterator = generator()
        else:
            raise ValueError("Unsupported type for `images`")

        if star_mask is None:
            def generator():
                while True:
                    yield None
            star_mask_iterator = generator()
        elif isinstance(star_mask, Generator):
            star_mask_iterator = star_mask
        elif isinstance(star_mask, np.ndarray):
            if len(star_mask.shape) == 3:
                def generator():
                    for mask in star_mask:
                        yield mask
                star_mask_iterator = generator()
            else:
                raise ValueError("Star mask array must be 3D")
        elif isinstance(star_mask, List) and isinstance(star_mask[0], str):
            def generator():
                for mask_path in star_mask:
                    with fits.open(mask_path) as hdul:
                        yield hdul[hdu_choice].data.astype(bool)
            star_mask_iterator = generator()
        else:
            raise ValueError("Unsupported type for `star_mask`")

        # the output collection to return
        this_collection = cls({})

        # We'll store the first image's shape, and then make sure the others
        # match.
        image_shape = None

        # for each image do the magic
        for i, (image, star_mask) in enumerate(zip(data_iterator, star_mask_iterator)):
            if image_shape is None:
                image_shape = image.shape
            elif image.shape != image_shape:
                msg = ("Images must all be the same shape."
                      f"Found both {image_shape} and {image.shape}.")
                raise ValueError(msg)

            # if the image should be scaled then, do the scaling before anything else
            if interpolation_scale != 1:
                interpolator = RectBivariateSpline(np.arange(image.shape[0]), 
                                                   np.arange(image.shape[1]), 
                                                   image)
                image = interpolator(np.linspace(0,
                                                 image.shape[0] - 1,
                                                 1 + (image.shape[0] - 1) * interpolation_scale),
                                     np.linspace(0,
                                                 image.shape[1] - 1,
                                                 1 + (image.shape[1] - 1) * interpolation_scale))

            # find stars using SEP
            background = sep.Background(image)
            image_background_removed = image - background
            image_star_coords = sep.extract(image_background_removed, 
                                            star_threshold, 
                                            err=background.globalrms,
                                            mask=star_mask)

            coordinates = [CoordinateIdentifier(i,
                                                int(round(y - psf_size * interpolation_scale / 2)),
                                                int(round(x - psf_size * interpolation_scale / 2)))
                           for x, y in zip(image_star_coords["x"], image_star_coords["y"])]

            # pad in case someone selects a region on the edge of the image
            padding_shape = ((psf_size * interpolation_scale, psf_size * interpolation_scale),
                             (psf_size * interpolation_scale, psf_size * interpolation_scale))
            padded_image = np.pad(image_background_removed,
                                  padding_shape, 
                                  mode="constant", 
                                  constant_values=np.median(image))

            for coordinate in coordinates:
                patch = padded_image[coordinate.x+interpolation_scale*psf_size:
                                     coordinate.x+2*interpolation_scale*psf_size,
                                     coordinate.y + interpolation_scale * psf_size:
                                     coordinate.y + 2 * interpolation_scale * psf_size]
                this_collection.add(coordinate, patch)

        corners = calculate_covering((image_shape[0] * interpolation_scale, 
                                      image_shape[1] * interpolation_scale),
                                      patch_size * interpolation_scale)
        averaged = this_collection.average(corners, 
                                           patch_size * interpolation_scale, psf_size * interpolation_scale,
                                           mode=average_mode, percentile=percentile)

        if interpolation_scale != 1:
            for coordinate, _ in averaged.items():
                averaged.patches[coordinate] = downscale_local_mean(averaged.patches[coordinate],
                                                                    (interpolation_scale, interpolation_scale))

            averaged.size = psf_size

        output = CoordinatePatchCollection({}, counts={})
        for key, patch in averaged.items():
            count = averaged.counts[key]
            output.add(CoordinateIdentifier(key.image_index,
                                            key.x // interpolation_scale,
                                            key.y // interpolation_scale),
                       patch,
                       count=count)

        return output

    def average(self, corners: np.ndarray, patch_size: int, psf_size: int,  # noqa: ARG002, kept for consistency
                mode: str = "median", percentile: float = 10) -> PatchCollectionABC:
        CoordinatePatchCollection._validate_average_mode(mode, percentile)

        if mode == "mean":
            mean_stack = {tuple(corner): np.zeros((psf_size, psf_size))
                          for corner in corners}
            mean_counts = {tuple(corner): np.zeros((psf_size, psf_size))
                          for corner in corners}
        else:
            # n.b. If mode is 'median', we could set mode='percentile'
            # and percentile=50 to simplify parts of this function, but
            # np.nanpercentile(x, 50) seems to be about half as fast as
            # np.nanmedian(x), so let's keep a speedy special case for medians.
            stack = {tuple(corner): [] for corner in corners}
        counts = {tuple(corner): 0 for corner in corners}

        corners_x, corners_y = corners[:, 0], corners[:, 1]
        x_bounds = np.stack([corners_x, corners_x + patch_size], axis=-1)
        y_bounds = np.stack([corners_y, corners_y + patch_size], axis=-1)

        for identifier, patch in self.patches.items():
            # Normalize the patch
            patch = patch / patch[psf_size//2, psf_size//2]

            # Determine which average region it belongs to
            center_x = identifier.x + self.size // 2
            center_y = identifier.y + self.size // 2
            x_matches = (x_bounds[:, 0] <= center_x) * (center_x < x_bounds[:, 1])
            y_matches = (y_bounds[:, 0] <= center_y) * (center_y < y_bounds[:, 1])
            match_indices = np.where(x_matches * y_matches)[0]

            # add to averages and increment count
            for match_index in match_indices:
                match_corner = tuple(corners[match_index])
                if mode == "mean":
                    mean_stack[match_corner] = np.nansum([mean_stack[match_corner], 
                                                          patch], axis=0)
                    mean_counts[match_corner] += np.isfinite(patch)
                else:
                    stack[match_corner].append(patch)
                counts[match_corner] += 1

        if mode == "mean":
            averages = {CoordinateIdentifier(None, corner[0], corner[1]): 
                        mean_stack[corner] / mean_counts[corner]
                        for corner in mean_stack}
        elif mode == "median":
            averages = {CoordinateIdentifier(None, corner[0], corner[1]):
                            np.nanmedian(stack[corner], axis=0)
                                if len(stack[corner]) > 0 else
                                np.zeros((psf_size, psf_size))
                        for corner in stack}
        elif mode == "percentile":
            averages = {CoordinateIdentifier(None, corner[0], corner[1]):
                            np.nanpercentile(stack[corner],
                                             percentile,
                                             axis=0)
                                if len(stack[corner]) > 0 else
                                np.zeros((psf_size, psf_size))
                        for corner in stack}
        counts = {CoordinateIdentifier(None, corner[0], corner[1]): count
                  for corner, count in counts.items()}
        # Now that we have our combined patches, pad them as appropriate
        pad_shape = self._calculate_pad_shape(patch_size)
        for key, patch in averages.items():
            averages[key] = np.pad(patch, pad_shape, mode="constant")
        return CoordinatePatchCollection(averages, counts=counts)

    @staticmethod
    def _validate_average_mode(mode: str, percentile: float) -> None:
        """Determine if the average_mode is a valid kind"""
        valid_modes = ["median", "mean", "percentile"]
        if mode not in valid_modes:
            msg = f"Found a mode of {mode} but it must be in the list {valid_modes}."
            raise ValueError(msg)
        if mode == "percentile" and not (0 <= percentile <= 100):
            raise ValueError("`percentile` must be between 0 and 100, inclusive")

    def _calculate_pad_shape(self, size: int) -> Tuple[int, int]:
        pad_amount = size - self.size
        if pad_amount < 0:
            raise InvalidSizeError(f"The average window size (found {size})" 
                                   "must be larger than the existing patch size"
                                   f"(found {self.size}).")
        if pad_amount % 2 != 0:
            raise InvalidSizeError(f"The average window size (found {size})" 
                                   "must be the same parity as the existing patch size"
                                   f"(found {self.size}).")
        return ((pad_amount//2, pad_amount//2), (pad_amount//2, pad_amount//2))

    def fit(self, base_psf: SimplePSF, 
            is_varied: bool = False) -> PointSpreadFunctionABC:
        raise NotImplementedError("TODO")

    def to_array_corrector(self, target_evaluation: np.array) -> ArrayCorrector:
        """Converts a patch collection that has been averaged into an ArrayCorrector

        Parameters
        ----------
        target_evaluation : np.ndarray
            the evaluation of the Target PSF

        Returns
        -------
        ArrayCorrector
            An array corrector that can be used to correct PSFs
        """
        evaluation_dictionary = {}
        for identifier, patch in self.patches.items():
            corrected_patch = patch.copy()
            corrected_patch[np.isnan(corrected_patch)] = 0
            evaluation_dictionary[(identifier.x, identifier.y)] = corrected_patch

        return ArrayCorrector(evaluation_dictionary, target_evaluation)

    def save(self, path: str) -> None:
        """Save the CoordinatePatchCollection to a file

        Parameters
        ----------
        path : str
            where to save the patch collection

        Returns
        -------
        None
        """
        with h5py.File(path, 'w') as f:
            patch_grp = f.create_group('patches')
            for key, val in self.patches.items():
                patch_grp.create_dataset(f"({key.image_index, key.x, key.y})", data=val)

    @classmethod
    def load(cls, path: str) -> PatchCollectionABC:
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
        patches = dict()
        with h5py.File(path, "r") as f:
            for key, val in f['patches'].items():
                parsed_key = tuple(int(val) for val in key.replace("(", "").replace(")", "").split(","))
                coord_id = CoordinateIdentifier(image_index=parsed_key[0], x=parsed_key[1], y=parsed_key[2])
                patches[coord_id] = val[:].copy()
        return cls(patches)
