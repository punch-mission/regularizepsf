from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import dill
import numpy as np
import deepdish as dd

from regularizepsf.exceptions import InvalidSizeError, EvaluatedModelInconsistentSizeError, UnevaluatedPointError
from regularizepsf.psf import VariedPSF, SimplePSF, PointSpreadFunctionABC
from regularizepsf.helper import _correct_image, _precalculate_ffts


class CorrectorABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def save(self, path: str | Path) -> None:
        """Save the model to a file.

        Parameters
        ----------
        path : str or `pathlib.Path`
            where to save the model, suggested extension is ".corr"

        Returns
        -------
        None
        """

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str | Path) -> CorrectorABC:
        """Loads a model from the path

        Parameters
        ----------
        path : str or `pathlib.Path`
            where to load the model from, suggested extension is ".corr"

        Returns
        -------

        """

    @abc.abstractmethod
    def correct_image(self, image: np.ndarray, size: int = None,
                      alpha: float = 0.5, epsilon: float = 0.05, use_gpu: bool = False) -> np.ndarray:
        """PSF correct an image according to the model

        Parameters
        ----------
        image : 2D float np.ndarray
            image to be corrected
        size : int
            how big to make the patches when correcting an image, only used for FunctionalCorrector
        alpha : float
            controls the “hardness” of the transition from amplification to attenuation, see notes
        epsilon : float
            controls the maximum of the amplification, see notes
        use_gpu : bool
            True uses GPU acceleration, False does not.

        Returns
        -------
        np.ndarray
            a image that has been PSF corrected

        Notes
        -----
        # TODO: add notes
        """


class FunctionalCorrector(CorrectorABC):
    """
    A version of the PSF corrector that stores the model as a set of functions.
    For the actual correction, the functions must first be evaluated to an ArrayCorrector.
    """
    def __init__(self, psf: PointSpreadFunctionABC,
                 target_model: SimplePSF | None):
        """Initialize a FunctionalCorrector

        Parameters
        ----------
        psf : SimplePSF or VariedPSF
            the model describing the psf for each patch of the image
        target_model : SimplePSF or None
            the target PSF to use to establish uniformity across the image
        """
        self._psf: PointSpreadFunctionABC = psf
        self._variable: bool = isinstance(self._psf, VariedPSF)
        self._target_model: SimplePSF = target_model

    @property
    def is_variable(self) -> bool:
        """
        Returns
        -------
        bool
            True if the PSF model is varied (changes across the field-of-view) and False otherwise
        """
        return self._variable

    def evaluate_to_array_form(self, x: np.ndarray, y: np.ndarray, size: int) -> ArrayCorrector:
        """Evaluates a FunctionalCorrector to an ArrayCorrector

        Parameters
        ----------
        x : np.ndarray
            the first dimension coordinates to evaluate over
        y : np.ndarray
            the second dimension coordinates to evaluate over
        size : int
            how large the patches in the PSF correction model shouuld be

        Returns
        -------
        ArrayCorrector
            an array evaluated form of this PSF corrector
        """
        if size % 2 != 0:
            raise InvalidSizeError(f"size must be even. Found size={size}.")

        image_x, image_y = np.meshgrid(np.arange(size), np.arange(size))
        evaluations = dict()
        for xx in x:
            for yy in y:
                evaluations[(xx, yy)] = self._psf(image_x, image_y)

        target_evaluation = self._target_model(image_x, image_y)
        return ArrayCorrector(evaluations, target_evaluation)

    def correct_image(self, image: np.ndarray, size: int = None,
                      alpha: float = 0.5, epsilon: float = 0.05, use_gpu: bool = False) -> np.ndarray:
        corners = calculate_covering(image.shape, size)
        array_corrector = self.evaluate_to_array_form(corners[:, 0], corners[:, 1], size)
        return array_corrector.correct_image(image, size=size, alpha=alpha, epsilon=epsilon, use_gpu=use_gpu)

    def save(self, path):
        with open(path, 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return dill.load(f)


class ArrayCorrector(CorrectorABC):
    """ A PSF corrector that is evaluated as array patches
    """
    def __init__(self, evaluations: dict[Any, np.ndarray], target_evaluation: np.ndarray):
        """Initialize an ArrayCorrector

        Parameters
        ----------
        evaluations : dict
            evaluated version of the PSF as they vary over the image, keys should be (x, y) of the lower left
                pixel of each patch. values should be the `np.ndarray` that corresponds to that patch
        target_evaluation : np.ndarray
            evaluated version of the target PSF
        """
        self._evaluation_points: list[Any] = list(evaluations.keys())

        if len(evaluations[self._evaluation_points[0]].shape) != 2:
            raise InvalidSizeError(f"PSF evaluations must be 2-D numpy arrays.")
        self._size = evaluations[self._evaluation_points[0]].shape[0]
        if self._size <= 0:
            raise InvalidSizeError(f"Found size of {self._size}. Must be >= 1")
        if self._size % 2 != 0:
            raise InvalidSizeError(f"Size must be even. Found {self._size}")

        self._evaluations: dict[Any, np.ndarray] = evaluations
        for (x, y), evaluation in self._evaluations.items():
            if evaluation.shape != (self._size, self._size):
                raise EvaluatedModelInconsistentSizeError(f"Expected evaluated model to have shapes of "
                                                          f"{(self._size, self._size)}. Found {evaluation.shape} "
                                                          f"at {(x, y)}.")

        self._target_evaluation = target_evaluation
        if self._target_evaluation.shape != (self._size, self._size):
            raise EvaluatedModelInconsistentSizeError("The target and evaluations must have the same shape.")

        values = np.array([v for v in self._evaluations.values()], dtype=float)
        self.target_fft, self.psf_i_fft = _precalculate_ffts(self._target_evaluation, values)

    def correct_image(self, image: np.ndarray, size: int = None,
                      alpha: float = 0.5, epsilon: float = 0.05, use_gpu: bool = False) -> np.ndarray:
        if not all(img_dim_i >= psf_dim_i for img_dim_i, psf_dim_i in zip(image.shape, (self._size, self._size))):
            raise InvalidSizeError("The image must be at least as large as the PSFs in all dimensions")

        x = np.array([x for x, _ in self._evaluations.keys()], dtype=int)
        y = np.array([y for _, y in self._evaluations.keys()], dtype=int)

        return _correct_image(image, self.target_fft, x, y, self.psf_i_fft,  alpha, epsilon)

    def __getitem__(self, xy) -> np.ndarray:
        if xy in self._evaluation_points:
            return self._evaluations[xy]
        else:
            raise UnevaluatedPointError(f"Model not evaluated at {xy}.")

    def save(self, path):
        dd.io.save(path, (self._evaluations, self._target_evaluation))

    @classmethod
    def load(cls, path):
        evaluations, target_evaluation = dd.io.load(path)
        return cls(evaluations, target_evaluation)


def calculate_covering(image_shape: tuple[int, int], size: int) -> np.ndarray:
    """Determines the grid of patches to sum over.

    Parameters
    ----------
    image_shape : tuple of 2 ints
        shape of the image we plan to correct
    size : int
        size of the square patches we want to create

    Returns
    -------
    np.ndarray
        an array of shape Nx2 where return[:, 0] are the x coordinate and return[:, 1] are the y coordinates

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
