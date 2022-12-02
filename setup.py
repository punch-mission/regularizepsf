from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

ext_modules = [Extension('psfpy.helper',
                         sources=['psfpy/helper.pyx'],
                         include_dirs=[numpy.get_include()])]

setup(
    name='psfpy',
    version='0.0.1',
    description='Point spread function modeling and correction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['psfpy'],
    url='https://github.com/punch-mission/psfpy',
    license='MIT',
    author='J. Marcus Hughes',
    author_email='hughes.jmb@gmail.com',
    ext_modules=cythonize(ext_modules, annotate=True, compiler_directives={'language_level': 3}),
    install_requires=["numpy", "dill", "deepdish", "lmfit", "sep", "cython", "astropy", "scipy",
                      "photutils", "scikit-image"],
    package_data={"psfpy": ["helper.pyx"]},
    setup_requires=["cython"],
    extras_require={"test": ['pytest', 'coverage', 'pytest-runner']}
)
