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

SHOW_FIGURES = False
patch_size, psf_size = 700, 32
out_dir = "/Users/jhughes/Desktop/projects/PUNCH/psf_paper"
fn = "/Users/jhughes/Nextcloud/PSF/DASH_2014-07-22T22:37:51.040_LF_expTime_10_numInBurst_8_ccdTemp_-20.0046.fits"

with fits.open(fn) as hdul:
    header = hdul[0].header
    data = hdul[0].data.astype(float)


if SHOW_FIGURES:
    m, s = np.mean(data), np.std(data)
    fig, ax = plt.subplots()
    im = ax.imshow(data, interpolation='nearest', cmap='gray', vmin=m-s, vmax=m+s, origin='lower')
    fig.colorbar(im)
    fig.show()

bkg = sep.Background(data)
bkg_image = bkg.back()

if SHOW_FIGURES:
    fig, ax = plt.subplots()
    im = ax.imshow(bkg_image, interpolation='nearest', cmap='gray', origin='lower')
    fig.colorbar(im)
    fig.show()

bkg_rms = bkg.rms()

if SHOW_FIGURES:
    fig, ax = plt.subplots()
    im = ax.imshow(bkg_rms, interpolation='nearest', cmap='gray', origin='lower')
    fig.colorbar(im)
    fig.show()

data_sub = data - bkg
objects = sep.extract(data_sub,3, err=bkg.globalrms)
d = data_sub

if SHOW_FIGURES:
    # plot background-subtracted image
    fig, ax = plt.subplots()
    m, s = np.mean(d), np.std(d)
    im = ax.imshow(d, interpolation='nearest', cmap='gray',
                   vmin=m - s, vmax=m + s, origin='lower')

    # plot an ellipse for each object
    for i in range(len(objects)):
        e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                    width=6 * objects['a'][i],
                    height=6 * objects['b'][i],
                    angle=objects['theta'][i] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)
    plt.show()

coordinates = [CoordinateIdentifier(0, int(x)-psf_size//2, int(y)-psf_size//2) for y, x in zip(objects['x'], objects['y'])]
patch_collection = CoordinatePatchCollection.extract([d], coordinates, psf_size)

if SHOW_FIGURES:
    i = 1652

    patch_identifiers = list(patch_collection.keys())
    fig, ax = plt.subplots()
    ax.imshow(patch_collection[patch_identifiers[i]])
    fig.show()

corners = calculate_covering(d.shape, patch_size)
averaged = patch_collection.average(corners, patch_size, psf_size)

if SHOW_FIGURES:
    i = 62

    averaged_identifiers = list(averaged.keys())

    patch = averaged[averaged_identifiers[i]]
    fig, ax = plt.subplots()
    im = ax.imshow(patch)
    fig.colorbar(im)
    fig.show()


@simple_psf
def dash_target(x, y, x0=patch_size/2, y0=patch_size/2, sigma_x=5/2.355, sigma_y=5/2.355):
    return np.exp(-(np.square(x-x0)/(2*np.square(sigma_x)) + np.square(y-y0)/(2*np.square(sigma_y))))


pad_amount = (patch_size - psf_size) // 2


evaluation_dictionary = dict()
for identifier, patch in averaged.items():
    corrected_patch = patch.copy()
    corrected_patch[np.isnan(corrected_patch)] = 0
    low, high = np.median(corrected_patch), np.nanpercentile(corrected_patch, 99.999) - 0.1
    corrected_patch = (corrected_patch - low) / (high - low)
    corrected_patch[corrected_patch < 0.05] = 0
    corrected_patch[corrected_patch > 0.95] = 1
    evaluation_dictionary[(identifier.x, identifier.y)] = np.pad(corrected_patch,
                                                                 ((pad_amount, pad_amount), (pad_amount, pad_amount)),
                                                                 mode='constant')

target_evaluation = dash_target(*np.meshgrid(np.arange(patch_size), np.arange(patch_size)))
array_corrector = ArrayCorrector(evaluation_dictionary, target_evaluation)

start = time.time()
corrected = array_corrector.correct_image(d, alpha=0.5, epsilon=0.05, apodize=True)
end = time.time()
print(end - start)

np.save(os.path.join(out_dir, "dash_uncorrected.npy"), d)
np.save(os.path.join(out_dir, "dash_corrected.npy"), corrected)
