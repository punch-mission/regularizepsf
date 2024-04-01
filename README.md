# regularizepsf
[![codecov](https://codecov.io/gh/punch-mission/regularizepsf/branch/main/graph/badge.svg?token=pn4NTO70I9)](https://codecov.io/gh/punch-mission/regularizepsf)
[![DOI](https://zenodo.org/badge/555583385.svg)](https://zenodo.org/badge/latestdoi/555583385)
[![PyPI version](https://badge.fury.io/py/regularizepsf.svg)](https://badge.fury.io/py/regularizepsf)
[![CI](https://github.com/punch-mission/regularizepsf/actions/workflows/ci.yml/badge.svg)](https://github.com/punch-mission/regularizepsf/actions/workflows/ci.yml)

A package for manipulating and correcting variable point spread functions.

Below is an example of correcting model data using the package. An initial image of a simplified starfield (a) is synthetically observed with a slowly
varying PSF (b), then regularized with this technique (c). The final image visually matches a direct convolution of
the initial image with the target PSF (d). The panels are gamma-corrected to highlight the periphery of the model PSFs.
![Example result image](model_example.png)

## Getting started

`pip install regularizepsf` and then follow along with the [documentation](https://punch-mission.github.io/regularizepsf/concepts.html).

## Contributing
We encourage all contributions. If you have a problem with the code or would like to see a new feature, please open an issue. Or you can submit a pull request.

If you're contributing code please see [this package's development guide](https://punch-mission.github.io/regularizepsf/development.html).

## License
See [LICENSE file](LICENSE)

## Need help?
Please ask a question in our [discussions](https://github.com/punch-mission/regularizepsf/discussions)

## Citation
Please cite [the associated paper](https://iopscience.iop.org/article/10.3847/1538-3881/acc578) if you use this technique:

```
@article{Hughes_2023,
doi = {10.3847/1538-3881/acc578},
url = {https://dx.doi.org/10.3847/1538-3881/acc578},
year = {2023},
month = {apr},
publisher = {The American Astronomical Society},
volume = {165},
number = {5},
pages = {204},
author = {J. Marcus Hughes and Craig E. DeForest and Daniel B. Seaton},
title = {Coma Off It: Regularizing Variable Point-spread Functions},
journal = {The Astronomical Journal}
}
```

If you use this software, please also cite the package with the specific version used. [Zenodo always has the most up-to-date citation](https://zenodo.org/records/10066960).
