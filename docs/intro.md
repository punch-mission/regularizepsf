# Welcome to PSFPy

`psfpy` is a Python package (with Cython speed improvements) for determining and correcting 
point spread functions in astronomical images.
It was originally developed for the [PUNCH](https://punch.space.swri.edu/) mission and is documented in this 
Astrophysical Journal paper. 

Below is an example of correcting PUNCH data using the package. Note that the residual tails in the corrected data are 
at or below about the 5\% stellar level. 
```{image} images/punch.png
:alt: example of correction
:width: 800px
:align: center
```
