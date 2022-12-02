import numpy as np
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time
import sep
from regularizepsf.fitter import CoordinatePatchCollection, CoordinateIdentifier
from regularizepsf.corrector import calculate_covering
from regularizepsf.psf import simple_psf
from regularizepsf.corrector import ArrayCorrector
from regularizepsf.fitter import CoordinatePatchCollection, CoordinateIdentifier


def main():
    patch_size = 32
    star_size = 2

    d = np.random.randint(-100, 100, (2048, 2048)).astype(float)
    # d = np.ones((2048, 2048)).astype(float)*500

    random_stars = np.random.randint(100, 1900, (500, 2))
    for x, y in random_stars:
        d[x:x+star_size, y:y+star_size] = np.random.randint(1000, 5000)
    corners = calculate_covering(d.shape, patch_size)
    averaged = CoordinatePatchCollection.extract([np.pad(np.ones((2048, 2048)), ((patch_size, patch_size), (patch_size, patch_size)))],
                                                 [CoordinateIdentifier(0, x, y) for x, y in corners], patch_size)
    averaged = CoordinatePatchCollection(dict())

    @simple_psf
    def ones_intermediate(x, y, x0=patch_size / 2, y0=patch_size / 2, sigma_x=2 / 2.355, sigma_y=2 / 2.355):
        out = np.zeros((patch_size, patch_size))#np.exp(-(np.square(x - x0) / (2 * np.square(sigma_x)) + np.square(y - y0) / (2 * np.square(sigma_y))))
        out[patch_size//2, patch_size//2] = 1
        return out

    for x, y in corners:
        averaged.add(CoordinateIdentifier(None, x, y), ones_intermediate(*np.meshgrid(np.arange(patch_size), np.arange((patch_size)))))

    @simple_psf
    def ones_target(x, y, x0=patch_size / 2, y0=patch_size / 2, sigma_x=3 / 2.355, sigma_y= 3/ 2.355):
        return np.exp(-(np.square(x - x0) / (2 * np.square(sigma_x)) + np.square(y - y0) / (2 * np.square(sigma_y))))


    evaluation_dictionary = dict()
    for identifier, patch in averaged.items():
        evaluation_dictionary[(identifier.x, identifier.y)] = ones_intermediate(*np.meshgrid(np.arange(patch_size), np.arange(patch_size)))

    target_evaluation = ones_target(*np.meshgrid(np.arange(patch_size), np.arange(patch_size)))
    array_corrector = ArrayCorrector(evaluation_dictionary, target_evaluation)

    print(f"Starting with {len(evaluation_dictionary)}")
    start = time.time()
    corrected = array_corrector.correct_image(d, alpha=0.5, epsilon=0.05)
    end = time.time()
    print(end - start)
    print("Finished!")

    np.save("noise_uncorrected.npy", d)
    np.save("noise_corrected.npy", corrected)


if __name__ == "__main__":
    main()
