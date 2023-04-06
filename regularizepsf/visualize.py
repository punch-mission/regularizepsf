import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from regularizepsf.corrector import ArrayCorrector
from regularizepsf.fitter import CoordinateIdentifier, PatchCollectionABC


def visualize_patch_counts(patch_collection: PatchCollectionABC,
                           ax: matplotlib.axes.Axes = None) -> matplotlib.axes.Axes:
    """
    Utility to visualize the number of stars identified within each patch

    Parameters
    ----------
    patch_collection : PatchCollectionABC
        A patch collection, such as that returned by
        `CoordinatePatchCollection.find_stars_and_average`.
    ax : matplotlib.axes.Axes
        An Axes object on which to plot. If not provided, a new Figure will be
        generated.
    """
    if patch_collection.counts is None or not len(patch_collection.counts):
        raise ValueError("This PatchCollection does not have any counts")

    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()

    corners = []
    counts = []
    for corner, count in patch_collection.counts.items():
        corners.append(corner)
        counts.append(count)
    # corners has the lower-left corner of each patch
    corners = np.array(corners)
    corners += patch_collection.size // 2
    # Now it has the center of each patch

    ax.set_aspect('equal')
    m = ax.scatter(corners[:, 1], corners[:, 0], c=counts, s=200)
    plt.colorbar(m).set_label(
            "Number of stars found in patch centered on point")

    return ax


def _generate_colormap():
    a = np.linspace(0, 1, 1000)
    r = np.sqrt(a)
    g = a
    b = np.square(a)
    colors = np.stack([r, g, b], axis=-1)
    return matplotlib.colors.ListedColormap(colors)


_colormap = _generate_colormap()


def visualize_PSFs(psfs: ArrayCorrector,
                   corrected: PatchCollectionABC = None,
                   all_patches: bool = False,
                   region_size: int = 0,
                   fig_scale: float = 1,
                   imshow_args: dict = {}) -> matplotlib.figure.Figure:
    """
    Utility to visualize computed PSFs.

    Accepts an `ArrayCorrector`, which contains the computed PSFs across the image.

    This utility can also produce a "before and after" visualization. To do
    this, apply your `ArrayCorrector` to your image set, and then run
    `CoordinatePatchCollection.find_stars_and_average` on your corrected
    images. This will compute the PSF of your corrected images. Pass the
    computed `CoordinatePatchCollection` as the `corrected` argument to this
    function.

    Parameters
    ----------
    psfs : ArrayCorrector
        An `ArrayCorrector` containing the computed PSFs
    corrected : PatchCollectionABC
        A `CoordinatePatchCollection` computed on the corrected set of images
    all_patches : boolean
        PSFs are computed for a grid of overlapping patches, with each image
        pixel being covered by four patches. If `True`, all of these patches
        are plotted, which can be useful for diagnosing the computed PSFs. If
        `False`, only a fourth of all patches are plotted (every other patch in
        both x and y), which can produce simpler illustrations.
    region_size : int
        The width of the central region of each patch to plot, or 0 to plot
        each entire patch. If the PSFs were computed with a `psf_size` less
        than `patch_size`, it may be convenient to set `region_size=psf_size`,
        to omit the empty edges of each patch.
    fig_scale : float
        Scale the image size up or down by this factor
    imshow_args : dict
        Additional arguments to pass to each `plt.imshow()` call

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Special-case vmin/vmax, and pass them to our PowerNorm
    vmin = imshow_args.pop('vmin', 0)
    vmax = imshow_args.pop('vmax', 1)
    imshow_args_default = dict(
        origin='lower',
        cmap=_colormap,
        norm=matplotlib.colors.PowerNorm(gamma=1/2.2, vmin=vmin, vmax=vmax)
    )
    imshow_args = imshow_args_default | imshow_args

    # Identify which patches we'll be plotting
    rows = np.unique(sorted(r for r, c in psfs._evaluation_points))
    columns = np.unique(sorted(c for r, c in psfs._evaluation_points))
    if not all_patches:
        rows = rows[1::2]
        columns = columns[1::2]

    if region_size:
        patch_size = psfs[rows[0], columns[0]].shape[0]
        trim = int((patch_size - region_size) / 2)
    else:
        trim = None

    # Work out the size of the image
    # Each grid of patches will be 6 inches wide
    patches_width = 6
    # Determine an image height based on the number of rows of patches
    patches_height = patches_width * len(rows) / len(columns)
    # Add space for the colorbar
    total_width = patches_width + .3
    # To make sure we have a little padding between the patches and the
    # colorbar, we'll add an extra, empty column
    n_columns = len(columns) + 2
    width_ratios = [patches_width/len(columns)]*len(columns) + [.1, .2]

    if corrected is not None:
        # Add space for a second grid of patches (including a padding column)
        total_width += patches_width + .2
        n_columns += len(columns) + 1
        width_ratios = (
                [patches_width / len(columns)] * len(columns)
                + [.2] + width_ratios)

    fig = matplotlib.figure.Figure(
            figsize=(total_width * fig_scale, patches_height * fig_scale))

    gs = matplotlib.gridspec.GridSpec(
            len(rows), n_columns, figure=fig,
            wspace=0, hspace=0,
            width_ratios=width_ratios)

    for i, j in itertools.product(range(len(rows)), range(len(columns))):
        ax = fig.add_subplot(gs[len(rows)-1-i, j])
        image = psfs[rows[i], columns[j]]
        if trim is not None:
            image = image[trim:-trim, trim:-trim]
        im = ax.imshow(image, **imshow_args)
        # Ensure there's a thin white line between subplots
        ax.spines[:].set_color('white')
        ax.set_xticks([])
        ax.set_yticks([])

    cax = fig.add_subplot(gs[:, -1])
    fig.colorbar(im, cax=cax, label='Normalized brightness')

    if corrected is not None:
        for i, j in itertools.product(range(len(rows)), range(len(columns))):
            ax = fig.add_subplot(gs[len(rows)-1-i, j + len(columns) + 1])
            image = corrected[CoordinateIdentifier(None, rows[i], columns[j])]
            if trim is not None:
                image = image[trim:-trim, trim:-trim]
            im = ax.imshow(image, **imshow_args)
            # Ensure there's a thin white line between subplots
            ax.spines[:].set_color('white')
            ax.set_xticks([])
            ax.set_yticks([])

        fig.text(0.31, 0.95, 'Uncorrected', ha='center', fontsize=15)
        fig.text(0.7, 0.95, 'Corrected', ha='center', fontsize=15)
    return fig

