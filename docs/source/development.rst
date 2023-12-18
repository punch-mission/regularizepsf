Development
============
We encourage all contributions. Please see our `contribution guide first <https://github.com/punch-mission/punch-mission/blob/main/contributing.md>`_.

We recommend working in a virtual environment.
This can be created by running ``python -m venv venv``. Then, activate the environment with ``source venv/bin/activate``.
You can then install the required packages with ``pip install -r requirements_dev.txt``.

If at any time you run into issues, please contact us by :doc:`following the guidelines here <help>`.

Building the docs
------------------
The docs are built using ``sphinx``. First, you must install it and the other documentation requirements with ::

    pip install -r ./docs/requirements.txt
    pip install -r requirements.txt
    python setup.py build_ext --inplace

Then, navigate to the ``docs`` directory and run ``make html`` to build the docs.

Running tests
-------------
To run the tests for this package, run ``pytest`` in the repository base directory.

This repository includes tests for the plotting utilities which compare generated plots to reference images saved in
``tests/baseline``.
To include these image-comparison tests, run ``pytest --mpl``.
To update these reference images, run ``pytest --mpl --mpl-generate-path=tests/baseline``.

If the image-comparison tests are failing,
run ``pytest --mpl --mpl-generate-summary=html`` to generate a summary page showing the generated and reference images.
The location of the generated file will be shown at the end of ``pytest``'s command-line output.
