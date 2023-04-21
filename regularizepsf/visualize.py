import copy
import itertools
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from regularizepsf.corrector import ArrayCorrector
from regularizepsf.fitter import CoordinateIdentifier, PatchCollectionABC
from regularizepsf.helper import _regularize_array


def visualize_patch_counts(patch_collection: PatchCollectionABC,
                           ax: Optional[matplotlib.axes.Axes] = None,
                           label_pixel_bounds: bool = False) -> matplotlib.axes.Axes:
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
    label_pixel_bounds : bool
        If True, the axes of the plot will be labeled with the pixel range
        spanned by each patch.
    """
    if patch_collection.counts is None or not len(patch_collection.counts):
        raise ValueError("This PatchCollection does not have any counts")

    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()

    rows = [k.x for k in patch_collection.counts.keys()]
    columns = [k.y for k in patch_collection.counts.keys()]
    rows = np.unique(sorted(rows))
    columns = np.unique(sorted(columns))
    dr = rows[1] - rows[0]
    dc = columns[1] - columns[0]

    # Build an array containing all the patch counts
    counts = np.empty((len(rows), len(columns)))
    for k, count in patch_collection.counts.items():
        r, c = k.x, k.y
        r = int((r - rows.min()) / dr)
        c = int((c - columns.min()) / dc)
        counts[r, c] = count

    m = ax.imshow(counts, origin='lower')
    plt.colorbar(m).set_label(
            "Number of stars found in patch")

    if label_pixel_bounds:
        xticks = [xt for xt in plt.xticks()[0] if 0 <= xt < len(columns)]
        plt.xticks(
                xticks,
                [f"{int(columns.min() + dc * i)}"
                 f" to\n{int(columns.min() + dc * (i+2))} px"
                    for i in xticks])
        yticks = [yt for yt in plt.yticks()[0] if 0 <= yt < len(rows)]
        plt.yticks(
                yticks,
                [f"{int(rows.min() + dr * i)}"
                 f" to\n{int(rows.min() + dr * (i+2))} px"
                    for i in yticks])
        ax.set_xlabel("Patch bounds (px)")
        ax.set_ylabel("Patch bounds (px)")
    else:
        ax.set_xlabel("Patch number")
        ax.set_ylabel("Patch number")

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
                   corrected: Optional[PatchCollectionABC] = None,
                   all_patches: bool = False,
                   region_size: int = 0,
                   label_pixel_bounds: bool = False,
                   fig: Optional[matplotlib.figure.Figure] = None,
                   fig_scale: float = 1,
                   colorbar_label: str = 'Normalized brightness',
                   axis_border_color: str = 'white',
                   imshow_args: dict = {}) -> matplotlib.figure.Figure:
    """
    Utility to visualize estimated PSFs.

    Accepts an `ArrayCorrector`, which contains the estimated PSFs across the
    image.

    This utility can also produce a "before and after" visualization. To do
    this, apply your `ArrayCorrector` to your image set, and then run
    `CoordinatePatchCollection.find_stars_and_average` on your corrected
    images. This will estimated the PSF of your corrected images. Pass this
    `CoordinatePatchCollection` as the `corrected` argument to this function.

    Parameters
    ----------
    psfs : ArrayCorrector
        An `ArrayCorrector` containing the estimated PSFs
    corrected : PatchCollectionABC
        A `CoordinatePatchCollection` computed on the corrected set of images
    all_patches : boolean
        PSFs are estimated in a grid of overlapping patches, with each image
        pixel being covered by four patches. If `True`, all of these patches
        are plotted, which can be useful for diagnosing the estimated PSFs. If
        `False`, only a fourth of all patches are plotted (every other patch in
        both x and y), which can produce simpler illustrations.
    region_size : int
        The width of the central region of each patch to plot, or 0 to plot
        each entire patch. If the PSFs were computed with a `psf_size` less
        than `patch_size`, it may be convenient to set `region_size=psf_size`,
        to omit the empty edges of each patch.
    label_pixel_bounds : bool
        If True, the axes of the plot will be labeled with the pixel range
        spanned by each patch.
    fig : matplotlb.figure.Figure
        A Figure on which to plot. If not provided, one will be created.
    fig_scale : float
        If `fig` is not provided, scale the generated Figure up or down by this
        factor.
    colorbar_label : str
        The label to show on the colorbar
    axis_border_color : str
        The color to use for the lines separating the patch plots.
    imshow_args : dict
        Additional arguments to pass to each `plt.imshow()` call

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Special-case vmin/vmax, and pass them to our PowerNorm
    if 'norm' not in imshow_args:
        vmin = imshow_args.pop('vmin', 0)
        vmax = imshow_args.pop('vmax', 1)
    else:
        # The default Norm will be overridden
        vmin, vmax = None, None
    imshow_args_default = dict(
        origin='lower',
        cmap=_colormap,
        norm=matplotlib.colors.PowerNorm(gamma=1/2.2, vmin=vmin, vmax=vmax)
    )
    imshow_args = imshow_args_default | imshow_args

    # Identify which patches we'll be plotting
    rows = np.unique(sorted(r for r, c in psfs._evaluation_points))
    columns = np.unique(sorted(c for r, c in psfs._evaluation_points))
    dr = rows[1] - rows[0]
    dc = columns[1] - columns[0]
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

    if fig is None:
        fig = plt.figure(
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
        # Ensure there's a thin line between subplots
        ax.spines[:].set_color(axis_border_color)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0 and label_pixel_bounds:
            ax.set_xlabel(
                    f"{int(columns[j])} to\n{int(columns[j] + 2 * dc)} px",
                    rotation='horizontal', ha='center')
        if j == 0 and label_pixel_bounds:
            ax.set_ylabel(
                    f"{int(rows[i])} to\n{int(rows[i] + 2 * dr)} px",
                    rotation='horizontal', ha='right', va='center')

    cax = fig.add_subplot(gs[:, -1])
    fig.colorbar(im, cax=cax, label=colorbar_label)

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


def visualize_transfer_kernels(psfs: ArrayCorrector,
                   alpha: float, epsilon: float,
                   all_patches: bool = False,
                   region_size: int = 0,
                   label_pixel_bounds: bool = False,
                   fig: Optional[matplotlib.figure.Figure] = None,
                   fig_scale: float = 1,
                   colorbar_label: str = 'Transfer kernel amplitude',
                   axis_border_color: str = 'black',
                   imshow_args: dict = {}) -> matplotlib.figure.Figure:
    """
    Utility to compute and visualize transfer kernels.

    Accepts an `ArrayCorrector`, which contains the estimated PSFs across the
    image. These PSFs, and the target PSF, will be used to compute each
    transfer kernel.

    Parameters
    ----------
    psfs : ArrayCorrector
        An `ArrayCorrector` containing the estimated PSFs and target PSF
    alpha, epsilon : float
        Values used in computing the regularized reciprocal of the computed
        PSFs. Provide the same values that you would pass to
        `ArrayCorrector.correct_image`.
    all_patches : boolean
        PSFs are estimated in a grid of overlapping patches, with each image
        pixel being covered by four patches. If `True`, all of these patches
        are plotted, which can be useful for diagnosing the computed PSFs. If
        `False`, only a fourth of all patches are plotted (every other patch in
        both x and y), which can produce simpler illustrations.
    region_size : int
        The width of the central region of each patch to plot, or 0 to plot
        each entire patch. If the PSFs were computed with a `psf_size` less
        than `patch_size`, it may be convenient to set `region_size=psf_size`,
        to omit the empty edges of each patch.
    label_pixel_bounds : bool
        If True, the axes of the plot will be labeled with the pixel range
        spanned by each patch.
    fig : matplotlb.figure.Figure
        A Figure on which to plot. If not provided, one will be created.
    fig_scale : float
        If `fig` is not provided, scale the generated Figure up or down by this
        factor.
    colorbar_label : str
        The label to show on the colorbar
    axis_border_color : str
        The color to use for the lines separating the patch plots.
    imshow_args : dict
        Additional arguments to pass to each `plt.imshow()` call

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Copy the ArrayCorrector. We'll change each patch to store the transfer
    # kernel rather than the PSF
    tks = copy.deepcopy(psfs)
    extent = -np.inf
    for i in range(len(tks._evaluations)):
        psf_regularized_inverse = _regularize_array(
                tks.psf_i_fft[i], alpha, epsilon, tks.target_fft)
        transfer_kernel = np.fft.ifft2(
                psf_regularized_inverse * tks.target_fft).real
        # For some reason, the transfer kernel we get straddles the corners of
        # the image, rather than being centered. We must be picking up a phase
        # shift in Fourier space.
        transfer_kernel = np.fft.fftshift(transfer_kernel)
        tks._evaluations[tks._evaluation_points[i]] = transfer_kernel
        extent = max(extent, np.max(np.abs(transfer_kernel)))

    # The plot we want is very similar to that produced by visualize_PSFs, so
    # we'll use that function and change some of the defaults.
    imshow_args_default = dict(
            norm=None,
            cmap='bwr',
            vmin=-extent,
            vmax=extent)
    imshow_args = imshow_args_default | imshow_args
    return visualize_PSFs(
            tks, all_patches=all_patches, region_size=region_size, fig=fig,
            fig_scale=fig_scale, colorbar_label=colorbar_label,
            axis_border_color=axis_border_color,
            label_pixel_bounds=label_pixel_bounds, imshow_args=imshow_args)
