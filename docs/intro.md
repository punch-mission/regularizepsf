# Welcome to regularizepsf

`regularizepsf` is a Python package (with Cython speed improvements) for determining and correcting 
point spread functions in astronomical images.
It was originally developed for the [PUNCH](https://punch.space.swri.edu/) mission and is documented in an upcoming 
Astrophysical Journal paper. For now, see [the arXiv listing](https://arxiv.org/abs/2212.02594). 

Below is an example of correcting model data using the package. An initial image of a simplified starfield (a) is synthetically observed with a slowly
varying PSF (b), then regularized with this technique (c). The final image visually matches a direct convolution of
the initial image with the target PSF (d). The panels are gamma-corrected to highlight the periphery of the model PSFs.
```{image} images/model.png
:alt: example of correction
:width: 800px
:align: center
```
