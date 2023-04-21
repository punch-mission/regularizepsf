---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python
  language: python3
  name: python3
---

# Development

## Building the docs
The docs are built using `jupyter-book`. With that package installed, open a terminal in the repository base directory and run `jupyter-book build docs`.

## Running tests
To run the tests for this package, run `pytest` in the repository base directory.

This repository includes tests for the plotting utilities which compare generated plots to reference images saved in `tests/baseline`. To include these image-comparison tests, run `pytest --mpl`. To update these reference images, run `pytest --mpl --mpl-generate-path=tests/baseline`.

If the image-comparison tests are failing, run `pytest --mpl --mpl-generate-summary=html` to generate a summary page showing the generated and reference images. The location of the generated file will be shown at the end of `pytest`'s command-line output.
