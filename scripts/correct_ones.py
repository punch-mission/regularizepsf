import numpy as np
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time
import sep
from psfpy.fitter import CoordinatePatchCollection, CoordinateIdentifier
from psfpy.corrector import calculate_covering
from psfpy.psf import simple_psf
from psfpy.corrector import ArrayCorrector
from psfpy.fitter import CoordinatePatchCollection, CoordinateIdentifier


def main():
    patch_size = 100

    d = np.ones((2048, 2048))
    corners = calculate_covering(d.shape, patch_size)
    averaged = CoordinatePatchCollection.extract([np.pad(d, ((patch_size, patch_size), (patch_size, patch_size)))],
                                                 [CoordinateIdentifier(0, x, y) for x, y in corners], patch_size)

    @simple_psf
    def ones_target(x, y, x0=patch_size / 2, y0=patch_size / 2, sigma_x=5 / 2.355, sigma_y=5 / 2.355):
        return np.ones((patch_size, patch_size)) #np.exp(-(np.square(x - x0) / (2 * np.square(sigma_x)) + np.square(y - y0) / (2 * np.square(sigma_y))))


    evaluation_dictionary = dict()
    for identifier, patch in averaged.items():
        evaluation_dictionary[(identifier.x, identifier.y)] = patch

    target_evaluation = ones_target(*np.meshgrid(np.arange(patch_size), np.arange(patch_size)))
    array_corrector = ArrayCorrector(evaluation_dictionary, target_evaluation)

    print(f"Starting with {len(evaluation_dictionary)}")
    start = time.time()
    corrected = array_corrector.correct_image(d, alpha=0.5, epsilon=0.05)
    end = time.time()
    print(end - start)
    print("Finished!")

    np.save("ones_uncorrected.npy", d)
    np.save("ones_corrected.npy", corrected)


if __name__ == "__main__":
    main()
