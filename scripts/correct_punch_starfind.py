from glob import glob
import os

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import sep

from psfpy.fitter import CoordinateIdentifier, CoordinatePatchCollection
from psfpy.corrector import ArrayCorrector, calculate_covering
from psfpy.psf import simple_psf


def get_punch_patch_collection(count=540,  psf_size=32, star_threshold: int = 3):
    half_size = psf_size // 2

    image_directory = ("/Users/jhughes/Nextcloud/23103_PUNCH_Data/"
                       "SOC_Data/PUNCH_WFI_EM_Starfield_campaign2_night2_phase3/calibrated/")
    image_filenames = sorted(glob(image_directory + "*.fits.gz"))

    patch_collection = CoordinatePatchCollection(dict())
    for i, fn in enumerate(image_filenames[:count]):
        with fits.open(fn) as hdul:
            img = hdul[0].data[26:-26, 50:-50].astype(float)

        background = sep.Background(img)
        image_background_removed = img - background
        image_star_coords = sep.extract(image_background_removed, star_threshold, err=background.globalrms)
        for x, y in zip(image_star_coords['x'], image_star_coords['y']):
            x, y = int(x), int(y)
            if 0 <= x < 2048 - psf_size and 0 <= y < 2048 - psf_size:
                patch = img[y - half_size:y + half_size, x - half_size:x + half_size]
                if patch.shape == (psf_size, psf_size):
                    patch_collection.add(CoordinateIdentifier(i, y, x), patch)
    patch_collection.save("punch_patch_collection_starfind_all.psfpy")


def average_patch_collection(psf_size=32, patch_size=400):
    all_stars = CoordinatePatchCollection.load("punch_patch_collection_starfind_all.psfpy")
    print(len(all_stars))
    corners = calculate_covering((2048, 2048), patch_size)
    averaged = all_stars.average(corners, patch_size, psf_size, mode='mean')
    averaged.save("punch_patch_collection_starfind_averaged.psfpy")


def convert_array_patch_collection_to_array_corrector(patch_size=400, psf_size=32):
    patch_collection = CoordinatePatchCollection.load("punch_patch_collection_starfind_averaged.psfpy")
    pad_amount = (patch_size - psf_size) // 2
    evaluation_dictionary = dict()
    for identifier, patch in patch_collection.items():
        corrected_patch = patch.copy()
        corrected_patch[np.isnan(corrected_patch)] = 0
        evaluation_dictionary[(identifier.x, identifier.y)] = np.pad(patch, ((pad_amount, pad_amount), (pad_amount, pad_amount)), mode='constant')

    @simple_psf
    def punch_target(x, y, x0=patch_size/2, y0=patch_size/2, sigma_x=2.25 / 2.355, sigma_y=2.25 / 2.355):
        return np.exp(-(np.square(x - x0) / (2 * np.square(sigma_x)) + np.square(y - y0) / (2 * np.square(sigma_y))))

    target_evaluation = punch_target(*np.meshgrid(np.arange(patch_size), np.arange(patch_size)))
    array_corrector = ArrayCorrector(evaluation_dictionary, target_evaluation)
    array_corrector.save("punch_array_corrector_starfind.psfpy")


if __name__ == "__main__":
    patch_size = 256
    # get_punch_patch_collection(count=540)
    average_patch_collection(patch_size=patch_size)
    convert_array_patch_collection_to_array_corrector(patch_size=patch_size)

    image_directory = ("/Users/jhughes/Nextcloud/23103_PUNCH_Data/"
                       "SOC_Data/PUNCH_WFI_EM_Starfield_campaign2_night2_phase3/calibrated/")
    image_filenames = sorted(glob(image_directory + "*.fits.gz"))

    ac = ArrayCorrector.load("punch_array_corrector_starfind.psfpy")

    print("starting correcting")
    for i in range(540):
        print(i)
        path = image_filenames[i]
        with fits.open(path) as hdul:
            image = hdul[0].data[26:-26, 50:-50]
        image = np.asfarray(image.byteswap().newbyteorder('='), 'float')
        corrected = ac.correct_image(image,  alpha=3.0, epsilon=0.3)
        np.save(f"/Users/jhughes/Desktop/projects/PUNCH/psf_paper/punch_corrections/starfind/{os.path.basename(path).replace('.fits.gz', '.npy')}", corrected)
