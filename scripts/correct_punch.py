from glob import glob
import pickle
import os
import time

from astropy.io import fits
import pandas as pd
import numpy as np


from regularizepsf.fitter import CoordinateIdentifier, CoordinatePatchCollection
from regularizepsf.corrector import calculate_covering, ArrayCorrector
from regularizepsf.psf import simple_psf


def create_array_psf(patch_size=400, psf_size=32):
    # read in the star list and create a set of coordinates to use
    star_list = pd.read_csv("/Users/jhughes/Desktop/projects/PUNCH/wfi_analysis/wfi/star_list_full_correctangles.csv")
    with open("/Users/jhughes/Desktop/projects/PUNCH/wfi_analysis/wfi/test_results_full_correctangles.pkl", 'rb') as f:
        results = pickle.load(f)
    coordinates = []
    for row in star_list.iterrows():
        row = row[1]
        for t in range(540):
            hip = int(row['HIP'])
            x, y = row[f'expected_pix_x_{t:03d}'], row[f'expected_pix_y_{t:03d}']
            if not (np.isnan(x) and np.isnan(y)):
                x, y = int(x), int(y)
                if 0 < x < 2048 and 0 < y < 2048:
                    this_result = results[hip][t]
                    if this_result:
                        offset_x = this_result.params['centroid_x'].value - 16
                        offset_y = this_result.params['centroid_y'].value - 16
                        coordinates.append((t, x + offset_x, y + offset_y))
    coordinates = np.array(coordinates).astype(int)

    # Create the CoordinatePatchCollection
    half_size = psf_size // 2
    image_directory = ("/Users/jhughes/Nextcloud/23103_PUNCH_Data/"
                       "SOC_Data/PUNCH_WFI_EM_Starfield_campaign2_night2_phase3/calibrated/")
    image_filenames = sorted(glob(image_directory + "*.fits.gz"))

    out = CoordinatePatchCollection(dict())

    for t in range(540):
        locs = np.where(coordinates[:, 0] == t)[0]
        with fits.open(image_filenames[t]) as hdul:
            this_image = hdul[0].data
        for _, x, y in coordinates[locs]:
            patch = this_image[y - half_size:y + half_size, x - half_size:x + half_size]
            out.add(CoordinateIdentifier(t, y, x), patch)

    out.save("punch_all_stars.regularizepsf")
    corners = calculate_covering((2048, 2048), patch_size)
    averaged = out.average(corners, patch_size, psf_size, mode='mean')
    averaged.save("punch_patch_collection.regularizepsf")


@simple_psf
def punch_target(x, y, x0=200, y0=200, sigma_x=2.25/2.355, sigma_y=2.25/2.355):
    return np.exp(-(np.square(x-x0)/(2*np.square(sigma_x)) + np.square(y-y0)/(2*np.square(sigma_y))))


def convert_array_patch_collection_to_array_corrector():
    patch_collection = CoordinatePatchCollection.load("punch_patch_collection.psfpy")

    evaluation_dictionary = dict()
    for identifier, patch in patch_collection.items():
        corrected_patch = patch.copy()
        corrected_patch[np.isnan(corrected_patch)] = 0
        evaluation_dictionary[(identifier.x, identifier.y)] = np.pad(patch, ((184, 184), (184, 184)), mode='constant')

    target_evaluation = punch_target(*np.meshgrid(np.arange(400), np.arange(400)))
    array_corrector = ArrayCorrector(evaluation_dictionary, target_evaluation)
    array_corrector.save("punch_array_corrector.regularizepsf")


def correct_image(image_filename: str, ac):
    with fits.open(image_filename) as hdul:
        image = hdul[0].data[26:-26, 50:-50].copy()

    # image = np.asfarray(image.newbyteorder("S"), dtype='float')
    image = np.asfarray(image.byteswap().newbyteorder('='), 'float')
    start = time.time()
    c = ac.correct_image(image, alpha=3.0, epsilon=0.3)
    end = time.time()
    print(f"took {end-start} seconds")

    return c


if __name__ == "__main__":
    # create_array_psf()

    convert_array_patch_collection_to_array_corrector()

    image_directory = ("/Users/jhughes/Nextcloud/23103_PUNCH_Data/"
                       "SOC_Data/PUNCH_WFI_EM_Starfield_campaign2_night2_phase3/calibrated/")
    image_filenames = sorted(glob(image_directory + "*.fits.gz"))

    array_corrector = ArrayCorrector.load("punch_array_corrector.psfpy")
    print("num patches", len(array_corrector._evaluations))
    for i in range(540):
        path = image_filenames[i]
        print(i, path)
        corrected = correct_image(path, array_corrector)
        # np.save(f"/Users/jhughes/Desktop/projects/PUNCH/psf_paper/punch_corrections/"
        #         f"{os.path.basename(path).replace('.fits.gz', '.npy')}", corrected)
        # np.save(f"/Users/jhughes/Desktop/"
        #         f"{os.path.basename(path).replace('.fits.gz', '.npy')}", corrected)
