Concepts
==========

For a very thorough and mathematical treatment of the technique see `our Astronomical Journal paper <https://iopscience.iop.org/article/10.3847/1538-3881/acc578>`_.

Overview of the technique
-------------------------
A point spread function (PSF) describes how the optical system spreads light from sources.
The basic premise of this technique is to model the point spread function of an imager using stars as our point sources.
Then, we calculate the inverse PSF and apply it. We could directly convolve the inverse PSF but convolutions are slow.
A convolution in the image is the same as multiplying in Fourier space, a much faster operation, so we do that instead.
This package supports defining a PSF transformation from any input PSF to any target PSF.

Since the PSF can vary across the image, we create many local models that apply only in smaller regions of the image.
These regions overlap so the correction is smooth without hard transition edges.

Overview of the package
------------------------
The package has a few main components:

1. Representations of PSFs in ``regularizepsf.psf``
2. A method of transforming a PSF from one to another in ``regularizepsf.transform``
3. Routines to model a PSF from data in ``regularizepsf.builder``
4. Extra visualization tools in ``regularizepsf.visualize``

PSFs can be represented in three ways:

1. `simple_functional_psf`: the PSF is described as a mathematical function that doesn't vary across an image
2. `varied_functional_psf`: the PSF is described as a mathematical function that varies across the image
3. `ArrayPSF`: the PSF is described using many small arrays, avoiding the need to find an expressive functional model

Using a set of images, we can extract `ArrayPSF`s directly and quickly correct an image.
