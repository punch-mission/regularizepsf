Concepts
==========

For a very thorough and mathematical treatment of the technique see `our Astronomical Journal paper <https://iopscience.iop.org/article/10.3847/1538-3881/acc578>`_.

Overview of the technique
-------------------------
A point spread function (PSF) describes how the optical system spreads light from sources.
The basic premise of this technique is to model the point spread function of an imager using stars as our point sources.
Then, we calculate the inverse PSF and apply it. We could directly convolve the inverse PSF but convolutions are slow.
A convolution in the image is the same as multiplying in Fourier space, a much faster operation, so we do that instead.
Instead of simply calculating the inverse PSF directly, we also include a *target PSF* model to make the resulting corrected
stars uniform. This target is typically a Gaussian as shown below in the :doc:`examples`.

Since the PSF can vary across the image, we create many local models that apply only in smaller regions of the image.
These regions overlap so the correction is smooth without hard transition edges.

Purpose of the target PSF
~~~~~~~~~~~~~~~~~~~~~~~~~~
We need a target PSF because we not only correct the PSF but homogenize to a standard PSF.
This is necessary because a correction without applying a target could result in different PSFs across the image,
e.g. you might correct the center of the image much better than the edges because the center was more well behaved to begin with.
For our purposes on the PUNCH mission, we need the resulting PSF to be the same everywhere.
Thus, we apply a standardized target PSF as well as inverting the observed PSF.

Overview of the package
------------------------
The package has a few main components:

1. Fitting routines in ``regularizepsf.fitter``
2. PSF correction tools in ``regularizepsf.corrector``
3. Visualization utilities in ``regularizepsf.visualize``

The most commonly used fitting routine is ``CoordinatePatchCollection.find_stars_and_average`` as demonstrated in :doc:`examples`.
This can then be converted to an ``ArrayCorrector``, the most commonly used PSF correction tool, by adding the target PSF.
This ``ArrayCorrector`` then can be applied to any new image using the associated ``correct_image`` method on it.
Finally, there are many visualizations to assist in understanding the results and derived PSF models. These are
outlined in :doc:`visualization`.

To speed up computation of the many FFTs, the correction is done in Cython in the ``helper.pyx`` file.
