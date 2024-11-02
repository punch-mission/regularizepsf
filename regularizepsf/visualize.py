"""Visualization tools for PSFs."""
import itertools

import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from regularizepsf.util import IndexedCube


def _generate_colormap() -> matplotlib.colors.ListedColormap:
    a = np.linspace(0, 1, 1000)
    r = np.sqrt(a)
    g = a
    b = np.square(a)
    colors = np.stack([r, g, b], axis=-1)
    return mpl.colors.ListedColormap(colors)


DEFAULT_COLORMAP = _generate_colormap()
PSF_IMSHOW_ARGS_DEFAULT = {
    "origin": "lower",
    "cmap": DEFAULT_COLORMAP,
    "norm": mpl.colors.PowerNorm(gamma=1 / 2.2, vmin=None, vmax=None),
}
KERNEL_IMSHOW_ARGS_DEFAULT = {
    "norm": None,
    "cmap": "bwr"
}


def visualize_patch_counts(counts: dict[tuple[int, int], int],
                           ax: mpl.axes.Axes | None = None,
                           label_pixel_bounds: bool = False) -> mpl.axes.Axes:
    """Visualize the number of stars identified within each patch.

    Parameters
    ----------
    counts : dict[tuple[int, int], int]
        The counts returned by an ArrayPSFBuilder.build
    ax : matplotlib.axes.Axes
        An Axes object on which to plot. If not provided, a new Figure will be
        generated.
    label_pixel_bounds : bool
        If True, the axes of the plot will be labeled with the pixel range
        spanned by each patch.

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()

    rows = [k[0] for k in counts.keys()]
    columns = [k[1] for k in counts.keys()]
    rows = np.unique(sorted(rows))
    columns = np.unique(sorted(columns))
    dr = rows[1] - rows[0]
    dc = columns[1] - columns[0]

    # Build an array containing all the patch counts
    counts_arr = np.empty((len(rows), len(columns)))
    for k, count in counts.items():
        r, c = k[0], k[1]
        r = int((r - rows.min()) / dr)
        c = int((c - columns.min()) / dc)
        counts_arr[r, c] = count

    m = ax.imshow(counts_arr, origin="lower")
    plt.colorbar(m).set_label("Number of stars found in patch")

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

def visualize_grid(data: IndexedCube,
                   second_data: IndexedCube | None = None,
                   title: str | tuple[str, str] = "",
                   fig: mpl.figure.Figure | None = None,
                   fig_scale: int = 1,
                   all_patches: bool = False,
                   imshow_args: dict | None = None,
                   colorbar_label: str  = "") -> None:  # noqa: ANN002, ANN003
    """Visualize the PSF model."""
    # Identify which patches we'll be plotting
    rows = np.unique(sorted(r for r, c in data.coordinates))
    columns = np.unique(sorted(c for r, c in data.coordinates))
    if not all_patches:
        rows = rows[1::2]
        columns = columns[1::2]

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
    width_ratios = [patches_width / len(columns)] * len(columns) + [.1, .2]

    if second_data is not None:
        # Add space for a second grid of patches (including a padding column)
        total_width += patches_width + .2
        n_columns += len(columns) + 1
        width_ratios = (
                [patches_width / len(columns)] * len(columns)
                + [.2] + width_ratios)

    if fig is None:
        fig = plt.figure(
            figsize=(total_width * fig_scale, patches_height * fig_scale))

    gs = mpl.gridspec.GridSpec(
        len(rows), n_columns, figure=fig,
        wspace=0, hspace=0,
        width_ratios=width_ratios)

    for i, j in itertools.product(range(len(rows)), range(len(columns))):
        ax = fig.add_subplot(gs[len(rows) - 1 - i, j])
        im = ax.imshow(data[rows[i], columns[j]], **imshow_args)
        # Ensure there's a thin line between subplots
        ax.spines[:].set_color("white")
        ax.set_xticks([])
        ax.set_yticks([])

    cax = fig.add_subplot(gs[:, -1])
    fig.colorbar(im, cax=cax, label=colorbar_label)

    if second_data is not None:
        for i, j in itertools.product(range(len(rows)), range(len(columns))):
            ax = fig.add_subplot(gs[len(rows) - 1 - i, j + len(columns) + 1])
            image = second_data[(rows[i], columns[j])]
            im = ax.imshow(image, **imshow_args)
            ax.spines[:].set_color("white")
            ax.set_xticks([])
            ax.set_yticks([])

        fig.text(0.31, 0.95, title[0], ha="center", fontsize=15)
        fig.text(0.7, 0.95, title[1], ha="center", fontsize=15)

    return fig
