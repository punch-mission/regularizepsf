from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

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
    description='Point spread function modeling and correction',
    ext_modules=cythonize(ext_modules, annotate=True, compiler_directives={'language_level': 3}),
    install_requires=["numpy", "dill", "deepdish", "lmfit", "sep", "cython", "astropy", "scipy",
                      "photutils", "scikit-image"],
    setup_requires=["cython"],
    extras_require={"test": ['pytest', 'coverage', 'pytest-runner']}
)
