Visualization
===================

``regularizepsf`` includes a handful of plotting utilities to help understand the results of your PSF fitting.

Distribution of identified stars
----------------------------------

Once stars have been identified by ``CoordinatePatchCollection.find_stars_and_average``,
the distribution of those stars across the image plane can be plotted.
This can help diagnose whether a sufficiently-large sample of stars is available in each patch---
if not, a larger patch size or more input images may be required. ::


    from regularizepsf import CoordinatePatchCollection, visualize_patch_counts
    psf_size, patch_size = 16, 256
    cpc = CoordinatePatchCollection.find_stars_and_average([image_fn], psf_size, patch_size)
    plt.figure(figsize=(8,5))
    visualize_patch_counts(cpc, ax=plt.gca())

.. figure:: ./images/star_distribution.png
    :alt: 2D histogram of stars identified across the image plane
    :width: 669px
    :align: center

    Two-dimensional histogram of identified star locations across the image plane,
    including stars from all provided input files.
    Note that each patch overlaps its neighboring patches---each star is counted in four separate patches.

Extracted PSFs
---------------

Once the ``CoordinatePatchCollection`` has been converted to an ``ArrayCorrector``, the estimated PSF for each patch can be visualized.
This can help diagnose whether the PSF estimation is robust, whether the image background is being adequately subtracted, etc.,
and also serves as a useful illustration of the degree of PSF variation in the instrument. ::


        from regularizepsf import CoordinatePatchCollection, visualize_PSFs
        psf_size, patch_size = 16, 256
        cpc = CoordinatePatchCollection.find_stars_and_average([image_fn], psf_size, patch_size)
        array_corrector = cpc.to_array_corrector(target_evaluation)

        plt.figure(figsize=(8,5))
        visualize_PSFs(array_corrector, region_size=psf_size, fig_scale=1.4)

.. figure:: ./images/estimated_psfs.png
    :alt: Display of estimated PSFs across the image plane
    :width: 754px
    :align: center

    Plot showing a number of estimated PSFs across the image plane.
    By default, only every other row and column of patches is shown,
    so that each displayed patch is more easily visible in a reasonable image size.
    The displayed position of each patch corresponds to the location of that patch in the image plane.

Extracted and Corrected PSFs
-----------------------------
A useful exercise is to plot the estimated and corrected PSFs side-by-side.
This can be done by applying the ``ArrayCorrector`` to each image,
and then repeating the star-finding and PSF-estimation step with the corrected images.
The resulting ``CoordinatePatchCollection`` is accepted as a second argument to ``visualize_PSFs``
to produce a side-by-side comparison.
In the following example, we will use a Python generator to produce the corrected images on-demand,
rather than saving them to disk and reading them back again. ::

    from regularizepsf import CoordinatePatchCollection, visualize_PSFs
    psf_size, patch_size = 16, 256
    cpc = CoordinatePatchCollection.find_stars_and_average([image_fn], psf_size, patch_size)
    array_corrector = cpc.to_array_corrector(target_evaluation)

    alpha = 6
    epsilon = .2
    images = [image_fn]

    def loader():
        for fname in images:
            image = fits.getdata(fname).astype(float)
            image = array_corrector.correct_image(image, alpha=alpha, epsilon=epsilon)
            yield image

    cpc_after = CoordinatePatchCollection.find_stars_and_average(loader(), psf_size, patch_size)

    visualize_PSFs(array_corrector, cpc_after, region_size=psf_size, fig_scale=1.2)

.. figure:: ./images/estimated_corrected_psfs.png
    :alt: Display of estimated and corrected PSFs across the image plane
    :width: 1233px
    :align: center

    Plot showing a number of estimated PSFs across the image plane, and the corresponding corrected PSFs.
    It can be seen that the corrected PSFs are much more uniform.

Transfer Kernels
-----------------
The *transfer kernel* for each patch is a convolution kernel that combines the effects of de-convolving the estimated
instrumental PSF and re-convolving with the target PSF.
While the actual corrections are done in Fourier space,
the effect of PSF regularization can be thought of as convolving each image patch with that patch's transfer kernel.
Visualizing these kernels may help identify the source of artifacts that appear in the corrected images.
Transfer kernels can be computed and plotted with a utility function::

    from regularizepsf import CoordinatePatchCollection, visualize_transfer_kernels

    psf_size, patch_size = 16, 256
    cpc = CoordinatePatchCollection.find_stars_and_average([image_fn], psf_size, patch_size)
    array_corrector = cpc.to_array_corrector(target_evaluation)

    alpha = 6
    epsilon = .2
    images = [image_fn]


    visualize.visualize_transfer_kernels(
        array_corrector,
        alpha,
        epsilon,
        region_size=psf_size,
        fig_scale=1.5,
        imshow_args=dict(vmin=-0.3, vmax=0.3))

.. figure:: ./images/transfer_kernels.png
    :alt: Display of transfer kernels across the image plane
    :width: 815px
    :align: center

    Plot showing transfer kernels across the image plane.
