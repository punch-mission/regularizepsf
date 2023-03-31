import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from regularizepsf.fitter import PatchCollectionABC


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


def visualize_patches(patch_collection: PatchCollectionABC,
                      ax: matplotlib.axes.Axes = None,
                      psf_size: int = None,
                      figsize: tuple[float, float] = None,
                      uniform_scaling: bool = False,
                      imshow_args: dict = {}) -> matplotlib.axes.Axes:
    """
    Utility to visualize the patches in a PatchCollection

    Parameters
    ----------
    patch_collection : PatchCollectionABC
        A patch collection, such as that returned by
        `CoordinatePatchCollection.find_stars_and_average`.
    ax : matplotlib.axes.Axes
        An Axes object on which to plot. If not provided, a new Figure will be
        generated.
    psf_size : int
        If the PatchCollection was generated using a PSF size smaller than the
        patch size, provide the psf_size here to trim the padding from each
        patch.
    figsize : tuple
        If `ax` is not provided, the size of the generated Figure can be set.
    uniform_scaling : boolean
        If True, use the scame colormap scaling for all patches. If False,
        normalize each one separately.
    imshow_args : dict
        Additional arguments to pass to each `plt.imshow()` call
    """
    if not patch_collection.patches:
        raise ValueError("This PatchCollection does not have any patches")

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()

    patch_size = patch_collection.size
    if psf_size is not None:
        trim = int((patch_size - psf_size) / 2)

    if uniform_scaling:
        # Work out vmin/vmax values that work for all patches
        vmin, vmax = np.inf, -np.inf
        for patch in patch_collection.values():
            if psf_size is not None and trim:
                patch = patch[trim:-trim, trim:-trim]
            vmin = min(vmin, patch.min())
            vmax = max(vmax, patch.max())
    else:
        vmin, vmax = None, None

    kwargs = {'vmin': vmin, 'vmax': vmax}
    kwargs.update(imshow_args)

    # Track the coordinates we see, to later set the overall plot bounds
    xs, ys = [], []

    for corner, patch in patch_collection.items():
        if psf_size is not None and trim:
            patch = patch[trim:-trim, trim:-trim]
        # Since the patches overlap each other, we need to plot each one so it
        # only reaches out to the point of overlap.
        extent = (corner.y + patch_size/4,
                  corner.y + 3*patch_size/4,
                  corner.x + patch_size/4,
                  corner.x + 3*patch_size/4)
        im = ax.imshow(patch,
                       origin='lower',
                       extent=extent,
                       **kwargs)
        xs.extend(extent[0:2])
        ys.extend(extent[2:])
    if uniform_scaling:
        # A colorbar only makes sense if every patch uses the same colormap
        plt.colorbar(im)
    ax.set_xlim(np.min(xs), np.max(xs))
    ax.set_ylim(np.min(ys), np.max(ys))
    return ax

