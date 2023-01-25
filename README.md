# regularizepsf
[![codecov](https://codecov.io/gh/punch-mission/regularizepsf/branch/main/graph/badge.svg?token=pn4NTO70I9)](https://codecov.io/gh/punch-mission/regularizepsf)
[![DOI](https://zenodo.org/badge/555583385.svg)](https://zenodo.org/badge/latestdoi/555583385)
[![PyPI version](https://badge.fury.io/py/regularizepsf.svg)](https://badge.fury.io/py/regularizepsf)

A package for manipulating and correcting variable point spread functions.

Below is an example of correcting model data using the package. An initial image of a simplified starfield (a) is synthetically observed with a slowly
varying PSF (b), then regularized with this technique (c). The final image visually matches a direct convolution of
the initial image with the target PSF (d). The panels are gamma-corrected to highlight the periphery of the model PSFs.
![Example result image](model_example.png)

## Getting started

`pip install regularizepsf` and then follow along with the [Quickstart section](https://punch-mission.github.io/regularizepsf/quickstart.html). 

## Contributing
We encourage all contributions. If you have a problem with the code or would like to see a new feature, please open an issue. Or you can submit a pull request. 

## License
See LICENSE for the MIT license

## Need help?
Please contact Marcus Hughes at [marcus.hughes@swri.org](mailto:marcus.hughes@swri.org).

## Citation
Please cite the associated paper if you use this technique: 

```
@misc{https://doi.org/10.48550/arxiv.2212.02594,
  doi = {10.48550/ARXIV.2212.02594},
  url = {https://arxiv.org/abs/2212.02594},
  author = {Hughes, J. M. and DeForest, C. E. and Seaton, D. B.},
  keywords = {Instrumentation and Methods for Astrophysics (astro-ph.IM), FOS: Physical sciences, FOS: Physical sciences},
  title = {Coma Off It: Removing Variable Point Spread Functions from Astronomical Images},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
