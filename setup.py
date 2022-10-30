from setuptools import setup, Extension
from numpy import get_include as np_get_include

setup(
    name='psfpy',
    version='0.0.1',
    packages=['psfpy'],
    url='',
    license='MIT',
    author='J. Marcus Hughes',
    author_email='hughes.jmb@gmail.com',
    description='Point spread function modeling and correction',
    ext_modules=[Extension('psfpy.helper', sources=['psfpy/helper.pyx'], include_dirs=[np_get_include()])]
)
