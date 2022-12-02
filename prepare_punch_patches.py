from glob import glob

import numpy as np
from astropy.io import fits

from psfpy.fitter import CoordinateIdentifier, CoordinatePatchCollection


if __name__ == "__main__":
    size = 32

    image_directory = ("/Users/jhughes/Nextcloud/23103_PUNCH_Data/"
                       "SOC_Data/PUNCH_WFI_EM_Starfield_campaign2_night2_phase3/calibrated/")
    image_filenames = sorted(glob(image_directory + "*.fits.gz"))

    coordinates = np.load("mcdonald_night2_phase3_coords.npy")

    out = CoordinatePatchCollection(dict())
    for image_index, image_filename in enumerate(image_filenames[:2]):
        with fits.open(image_filename) as hdul:
            this_image = hdul[0].data

        this_image_coords = coordinates[np.where(coordinates[:, 0] == image_index)]
        for coordinate in coordinates:
            j, i = int(coordinate[1]), int(coordinate[2])
            patch = this_image[i:i + size, j:j + size]
            out.add(CoordinateIdentifier(image_index, i, j), patch)

    print("saving")
    out.save("patches.psfpy")
