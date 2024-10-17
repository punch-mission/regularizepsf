"""Functions for building PSF models from images."""

import pathlib

from regularizepsf.psf import ArrayPSF


class ArrayPSFBuilder:
    """A builder that will take a series of images and construct an ArrayPSF to represent their implicit PSF."""

    def __init__(self) -> None:
        """Initialize an ArrayPSFBuilder."""

    def build(self, image_paths: list[pathlib.Path]) -> ArrayPSF:
        """Build the PSF model.

        Parameters
        ----------
        image_paths : list[pathlib.Path]
            paths of images to use

        Returns
        -------
        ArrayPSF
            an array PSF

        """
