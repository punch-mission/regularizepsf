import pathlib

from astropy.io import fits
import numpy as np
import pytest

import regularizepsf
from regularizepsf.fitter import CoordinatePatchCollection


TEST_DIR = pathlib.Path(__file__).parent.resolve()


@pytest.mark.mpl_image_compare(style='mpl20', remove_text=True)
def test_visualize_patch_counts():
    img_path = str(TEST_DIR / "data/DASH.fits")
    data = fits.getdata(img_path)
    # Cut down the data size so this test isn't so slow
    data = data[:800, :800].reshape((1, 800, 800)).astype(float)
    cpc = CoordinatePatchCollection.find_stars_and_average(
            data, 32, 100)

    ax = regularizepsf.visualize_patch_counts(cpc)
    # The remove_text option of pytest-mpl only removes tick labels, not axis
    # labels
    for ax in ax.figure.axes:
        ax.set_xlabel('')
        ax.set_ylabel('')
    return ax.figure


@pytest.mark.mpl_image_compare(style='mpl20', remove_text=True)
def test_visualize_PSFs():
    patch_size = 200
    psf_size = 24
    img_path = str(TEST_DIR / "data/DASH.fits")
    data = fits.getdata(img_path)
    # Cut down the data size so this test isn't so slow
    data = data[:800, :800].reshape((1, 800, 800)).astype(float)
    cpc = CoordinatePatchCollection.find_stars_and_average(
            data, psf_size, patch_size)

    ax = regularizepsf.visualize_PSFs(
            cpc.to_array_corrector(np.zeros((patch_size,patch_size))),
            cpc,
            region_size=psf_size)
    # The remove_text option of pytest-mpl only removes tick labels, not axis
    # labels
    for ax in ax.figure.axes:
        ax.set_xlabel('')
        ax.set_ylabel('')
    for text in list(ax.figure.texts):
        text.remove()
    return ax.figure


@pytest.mark.mpl_image_compare(style='mpl20', remove_text=True)
def test_visualize_transfer_kernels():
    patch_size = 200
    psf_size = 24
    img_path = str(TEST_DIR / "data/DASH.fits")
    data = fits.getdata(img_path)
    # Cut down the data size so this test isn't so slow
    data = data[:800, :800].reshape((1, 800, 800)).astype(float)
    cpc = CoordinatePatchCollection.find_stars_and_average(
            data, psf_size, patch_size)

    x, y = np.meshgrid(np.arange(patch_size), np.arange(patch_size))
    target_psf = np.exp(
            -(np.square(x - patch_size/2) / (2 * np.square(2))
              + np.square(y - patch_size/2) / (2 * np.square(2))))
    ax = regularizepsf.visualize_transfer_kernels(
            cpc.to_array_corrector(target_psf),
            region_size=psf_size,
            alpha=2, epsilon=.3,
            all_patches=True, fig_scale=1.3)
    # The remove_text option of pytest-mpl only removes tick labels, not axis
    # labels
    for ax in ax.figure.axes:
        ax.set_xlabel('')
        ax.set_ylabel('')
    for text in list(ax.figure.texts):
        text.remove()
    return ax.figure
