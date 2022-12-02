from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

ext_modules = [Extension('psfpy.helper',
                         sources=['psfpy/helper.pyx'],
                         include_dirs=[numpy.get_include()])]

setup(
    name='psfpy',
    version='0.0.1',
    packages=['psfpy'],
    url='',
    license='MIT',
    author='J. Marcus Hughes',
    author_email='hughes.jmb@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='Point spread function modeling and correction',
    ext_modules=cythonize(ext_modules, annotate=True, compiler_directives={'language_level': 3}),
    install_requires=["numpy", "dill", "deepdish", "lmfit", "sep", "cython", "astropy", "scipy",
                      "photutils", "scikit-image"],
    package_data={"psfpy": ["helper.pyx"]},
    setup_requires=["cython"],
    extras_require={"test": ['pytest', 'coverage', 'pytest-runner']}
)
