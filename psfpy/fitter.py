from __future__ import annotations

import abc
import warnings
from typing import Any
from collections import namedtuple

import numpy as np
import deepdish as dd

from psfpy.psf import SimplePSF, VariedPSF, PointSpreadFunctionABC
from psfpy.exceptions import InvalidSizeError


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
        """

        Parameters
        ----------
        images
        coordinates
        size

        Returns
        -------

        """

    def __getitem__(self, identifier) -> np.ndarray:
        """

        Parameters
        ----------
        identifier

        Returns
        -------

        """
        if identifier in self._patches:
            return self._patches[identifier]
        else:
            raise IndexError(f"{identifier} is not used to identify a patch in this collection.")

    def __contains__(self, identifier):
        """

        Parameters
        ----------
        identifier

        Returns
        -------

        """
        return identifier in self._patches

    def add(self, identifier, patch: np.ndarray) -> None:
        """

        Parameters
        ----------
        identifier
        patch

        Returns
        -------

        """
        if identifier in self._patches:
            # TODO: improve warning
            warnings.warn(f"{identifier} is being overwritten in this collection.", Warning)
        self._patches[identifier] = patch

        if self._size is None:
            self._size = patch.shape[0]
            # TODO : enforce square constraint

    @abc.abstractmethod
    def average(self, corners: np.ndarray, step: int, size: int) -> PatchCollectionABC:
        """

        Parameters
        ----------
        corners
        step
        size

        Returns
        -------

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

    def save(self, path):
        dd.io.save(path, self._patches)

    @classmethod
    def load(cls, path):
        return cls(dd.io.load(path))

    def keys(self):
        return self._patches.keys()

    def values(self):
        return self._patches.values()

    def items(self):
        return self._patches.items()

    def __next__(self):
        # TODO: implement
        pass

CoordinateIdentifier = namedtuple("CoordinateIdentifier", "image_index, x, y")


class CoordinatePatchCollection(PatchCollectionABC):
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

    def average(self, corners: np.ndarray, step: int, size: int) -> PatchCollectionABC:
        pad_amount = size - self._size
        if pad_amount < 0:
            raise InvalidSizeError(f"The average window size (found {size})" 
                                   f"must be larger than the existing patch size (found {self._size}).")
        if pad_amount % 2 != 0:
            raise InvalidSizeError(f"The average window size (found {size})" 
                                   f"must be the same parity as the existing patch size (found {self._size}).")
        pad_shape = ((pad_amount//2, pad_amount//2), (pad_amount//2, pad_amount//2))

        averages = {tuple(corner): np.zeros((size, size)) for corner in corners}
        counts = {tuple(corner): 0 for corner in corners}

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
                averages[match_corner] = np.nansum([averages[match_corner], padded_patch], axis=0)
                counts[match_corner] += 1

        averages = {CoordinateIdentifier(None, corner[0], corner[1]): averages[corner]/counts[corner]
                    for corner in averages}
        return CoordinatePatchCollection(averages)

    def fit(self, base_psf: SimplePSF, is_varied: bool = False) -> PointSpreadFunctionABC:
        raise NotImplementedError("TODO")  # TODO: implement
