from glob import glob

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

from regularizepsf.fitter import CoordinateIdentifier, CoordinatePatchCollection


def get_dash():
    fn = "/Users/jhughes/Nextcloud/PSF/DASH_2014-07-22T22:37:51.040_LF_expTime_10_numInBurst_8_ccdTemp_-20.0046.fits"
    with fits.open(fn) as hdul:
        data = hdul[0].data.astype(float)
    return [data]


def get_punch(count=10):
    image_directory = ("/Users/jhughes/Nextcloud/23103_PUNCH_Data/"
                       "SOC_Data/PUNCH_WFI_EM_Starfield_campaign2_night2_phase3/calibrated/")
    image_filenames = sorted(glob(image_directory + "*.fits.gz"))

    data = []
    for fn in image_filenames[:count]:
        with fits.open(fn) as hdul:
            img = hdul[0].data.astype(float)
        data.append(img)
    data = np.array(data)
    return data


if __name__ == "__main__":
    grid_size = 4
    imgs = get_punch()
    patch_collection = CoordinatePatchCollection.find_stars_and_create(imgs, 32)

    keys = list(patch_collection.keys())
    indices = np.random.choice(np.arange(len(patch_collection)), grid_size**2, replace=False)

    fig, axs = plt.subplots(ncols=grid_size, nrows=grid_size, sharex=True, sharey=True)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(patch_collection[keys[indices[i]]], origin='lower')
    fig.show()