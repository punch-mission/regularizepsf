"""Utility functions for regularizepsf."""

from __future__ import annotations

import numpy as np

from regularizepsf.exceptions import IncorrectShapeError, InvalidCoordinateError


def calculate_covering(image_shape: tuple[int, int], size: int) -> np.ndarray:
    """Determine the grid of overlapping neighborhood patches.

    Parameters
    ----------
    image_shape : tuple of 2 ints
        shape of the image we plan to correct
    size : int
        size of the square patches we want to create

    Returns
    -------
    np.ndarray
        an array of shape Nx2 where return[:, 0]
        are the x coordinate and return[:, 1] are the y coordinates

    """
    half_size = np.ceil(size / 2).astype(int)

    x1 = np.arange(0, image_shape[0], size)
    y1 = np.arange(0, image_shape[1], size)

    x2 = np.arange(-half_size, image_shape[0], size)
    y2 = np.arange(-half_size, image_shape[1], size)

    x3 = np.arange(-half_size, image_shape[0], size)
    y3 = np.arange(0, image_shape[1], size)

    x4 = np.arange(0, image_shape[0], size)
    y4 = np.arange(-half_size, image_shape[1], size)

    x1, y1 = np.meshgrid(x1, y1)
    x2, y2 = np.meshgrid(x2, y2)
    x3, y3 = np.meshgrid(x3, y3)
    x4, y4 = np.meshgrid(x4, y4)

    x1, y1 = x1.flatten(), y1.flatten()
    x2, y2 = x2.flatten(), y2.flatten()
    x3, y3 = x3.flatten(), y3.flatten()
    x4, y4 = x4.flatten(), y4.flatten()

    x = np.concatenate([x1, x2, x3, x4])
    y = np.concatenate([y1, y2, y3, y4])
    return np.stack([x, y], -1)


class IndexedCube:
    """A stack of arrays with assigned coordinates as keys."""

    def __init__(self, coordinates: list[tuple[int, int]], values: np.ndarray) -> None:
        """Initialize an IndexedCube.

        Parameters
        ----------
        coordinates : list[tuple[int, int]]
            list of image coordinates for upper left corner of the cube patches represented.
        values: np.ndarray
            an array of image cube patches, should be size (len(coordinates), x, y)
            where x and y are the dimensions of the patches

        """
        if len(values.shape) != 3:  # noqa: PLR2004
            msg = "Values must be three dimensional"
            raise IncorrectShapeError(msg)

        if len(coordinates) != values.shape[0]:
            msg = f"{len(coordinates)} coordinates defined but {values.shape[0]} values found."
            raise IncorrectShapeError(msg)

        self._coordinates = coordinates
        self._values = values

        self._index = {tuple(coordinate): i for i, coordinate in enumerate(self._coordinates)}

    @property
    def sample_shape(self) -> tuple[int, int]:
        """Shape of individual sample."""
        return self._values.shape[1], self._values.shape[2]

    def __getitem__(self, coordinate: tuple[int, int]) -> np.ndarray:
        """Get the sample associated with that coordinate.

        Parameters
        ----------
        coordinate: tuple[int, int]
            reference coordinate for requested array

        Returns
        -------
        np.ndarray
            sample at that coordinate

        """
        if coordinate not in self._index:
            msg = f"Coordinate {coordinate} not in TransferKernel."
            raise InvalidCoordinateError(msg)
        return self._values[self._index[coordinate]]

    def __setitem__(self, coordinate: tuple[int, int], value: np.ndarray) -> None:
        """Set the array associated with that coordinate.

        Parameters
        ----------
        coordinate: tuple[int, int]
            reference coordinate for sample

        value: np.ndarray
            value at the sample

        Returns
        -------
        np.ndarray
            sample array

        """
        if coordinate not in self._index:
            msg = f"Coordinate {coordinate} not in TransferKernel."
            raise InvalidCoordinateError(msg)

        if value.shape != self.sample_shape:
            msg = f"Cannot assign value of shape {value.shape} to transfer kernel of shape {self.sample_shape}."
            raise IncorrectShapeError(msg)

        self._values[self._index[coordinate]] = value

    @property
    def coordinates(self) -> list[tuple[int, int]]:
        """Retrieve coordinates the transfer kernel is defined on.

        Returns
        -------
        list[tuple[int, int]]
            coordinates the transfer kernel is defined on.

        """
        return self._coordinates

    @property
    def values(self) -> np.ndarray:
        """Retrieve values of the cube."""
        return self._values

    def __len__(self) -> int:
        """Return number of sample cube is indexed on.

        Returns
        -------
        int
            number of sample cube is indexed on.

        """
        return len(self.coordinates)

    def __eq__(self, other: IndexedCube) -> bool:
        """Test equality between two IndexedCubes."""
        if not isinstance(other, IndexedCube):
            msg = "Can only compare IndexedCube instances."
            raise TypeError(msg)
        return (
            self.coordinates == other.coordinates
            and self.sample_shape == other.sample_shape
            and np.allclose(self.values, other.values)
        )
