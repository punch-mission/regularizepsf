"""Tools to transform from one PSF to another."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import scipy

from regularizepsf.exceptions import InvalidCoordinateError
from regularizepsf.util import IndexedCube

if TYPE_CHECKING:
    import pathlib

    from regularizepsf.psf import ArrayPSF


class ArrayPSFTransform:
    """Representation of a transformation from a source to a target PSF that can be applied to images."""

    def __init__(self, transfer_kernel: IndexedCube) -> None:
        """Initialize a PSFTransform.

        Parameters
        ----------
        transfer_kernel: TransferKernel
            the transfer kernel required by this ArrayPSFTransform

        """
        self._transfer_kernel = transfer_kernel

    @property
    def psf_shape(self) -> tuple[int, int]:
        """Retrieve the shape of the individual PSFs for this transform."""
        return self._transfer_kernel.sample_shape

    @property
    def coordinates(self) -> list[tuple[int, int]]:
        """Retrieve the coordinates of the individual PSFs for this transform."""
        return self._transfer_kernel.coordinates

    def __len__(self) -> int:
        """Retrieve the number of coordinates used to represent this transform."""
        return len(self._transfer_kernel)

    @classmethod
    def construct(cls, source: ArrayPSF, target: ArrayPSF, alpha: float, epsilon: float) -> ArrayPSFTransform:
        """Construct an ArrayPSFTransform from a source to a target PSF.

        Parameters
        ----------
        source : ArrayPSF
            source point spread function
        target : ArrayPSF
            target point spread function
        alpha : float
            controls the “hardness” of the transition from amplification to attenuation
        epsilon : float
            controls the maximum of the amplification

        Returns
        -------
        ArrayPSFTransform
            corresponding ArrayPSFTransform instance

        """
        if np.any(np.array(source.coordinates) != np.array(target.coordinates)):
            msg = "Source PSF coordinates do not match target PSF coordinates."
            raise InvalidCoordinateError(msg)

        source_abs = abs(source.fft_evaluations)
        target_abs = abs(target.fft_evaluations)
        numerator = source.fft_evaluations.conjugate() * source_abs ** (alpha - 1)
        denominator = source_abs ** (alpha + 1) + (epsilon * target_abs) ** (alpha + 1)
        cube = IndexedCube(source.coordinates, (numerator / denominator) * target.fft_evaluations)
        return ArrayPSFTransform(cube)

    def apply(self, image: np.ndarray, workers: int | None = None, pad_mode: str = "symmetric") -> np.ndarray:
        """Apply the PSFTransform to an image.

        Parameters
        ----------
        image : np.ndarray
            image to apply the transform to
        workers: int | None
            Maximum number of workers to use for parallel computation of FFT.
            If negative, the value wraps around from os.cpu_count(). See scipy.fft.fft for more details.
        pad_mode: str
            how to pad the image when computing ffts, see np.pad for more details.

        Returns
        -------
        np.ndarray
            image with psf transformed

        """
        padded_image = np.pad(
            image,
            ((2 * self.psf_shape[0], 2 * self.psf_shape[0]), (2 * self.psf_shape[1], 2 * self.psf_shape[1])),
            mode=pad_mode,
        )

        def slice_padded_image(coordinate: tuple[int, int]) -> tuple[slice, slice]:
            """Get the slice objects for a coordinate patch in the padded cube."""
            row_slice = slice(
                coordinate[0] + self.psf_shape[0] * 2, coordinate[0] + self.psf_shape[0] + self.psf_shape[0] * 2,
            )
            col_slice = slice(
                coordinate[1] + self.psf_shape[1] * 2, coordinate[1] + self.psf_shape[1] + self.psf_shape[1] * 2,
            )
            return row_slice, col_slice

        row_arr, col_arr = np.meshgrid(np.arange(self.psf_shape[0]), np.arange(self.psf_shape[1]))
        apodization_window = np.sin((row_arr + 0.5) * (np.pi / self.psf_shape[0])) * np.sin(
            (col_arr + 0.5) * (np.pi / self.psf_shape[1]),
        )
        apodization_window = np.broadcast_to(apodization_window, (len(self), self.psf_shape[0], self.psf_shape[1]))

        patches = np.stack(
            [
                padded_image[slice_padded_image(coordinate)[0], slice_padded_image(coordinate)[1]]
                for coordinate in self.coordinates
            ],
        )
        patches = scipy.fft.fft2(apodization_window * patches, workers=workers)
        patches = np.real(scipy.fft.ifft2(patches * self._transfer_kernel.values, workers=workers))
        patches = patches * apodization_window

        reconstructed_image = np.zeros_like(padded_image)
        for coordinate, patch in zip(self.coordinates, patches, strict=True):
            reconstructed_image[slice_padded_image(coordinate)[0], slice_padded_image(coordinate)[1]] += patch

        return reconstructed_image[
            2 * self.psf_shape[0] : image.shape[0] + 2 * self.psf_shape[0],
            2 * self.psf_shape[1] : image.shape[1] + 2 * self.psf_shape[1],
        ]

    def visualize(self) -> None:
        """Visualize the PSFTransform.

        Returns
        -------
        None

        """

    def save(self, path: pathlib.Path) -> None:
        """Save a PSFTransform to a file.

        Parameters
        ----------
        path : pathlib.Path
            where to save the PSFTransform

        Returns
        -------
        None

        """
        with h5py.File(path, "w") as f:
            f.create_dataset("coordinates", data=self.coordinates)
            f.create_dataset("transfer_kernel", data=self._transfer_kernel.values)

    @classmethod
    def load(cls, path: pathlib.Path) -> ArrayPSFTransform:
        """Load a PSFTransform object.

        Parameters
        ----------
        path : pathlib.Path
            file to load the PSFTransform from

        Returns
        -------
        PSFTransform

        """
        with h5py.File(path, "r") as f:
            coordinates = [tuple(c) for c in f["coordinates"][:]]
            transfer_kernel = f["transfer_kernel"][:]
        kernel = IndexedCube(coordinates, transfer_kernel)
        return cls(kernel)

    def __eq__(self, other: ArrayPSFTransform) -> bool:
        """Test equality between two transforms."""
        if not isinstance(other, ArrayPSFTransform):
            msg = "Can only compare ArrayPSFTransform to another ArrayPSFTransform."
            raise TypeError(msg)
        return self._transfer_kernel == other._transfer_kernel
