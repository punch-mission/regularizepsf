from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [Extension('psfpy.helper',
                         sources=['psfpy/helper.pyx', 'psfpy/fftw_helper.c'],
                         include_dirs=['include', numpy.get_include()],
                         libraries=['m', 'fftw3'],
                         library_dirs=['lib'],
                         extra_compile_args=['-std=c99', '-O3'],
                         extra_link_args=['-Wl,-rpath,lib'])
               #           ),
               # Extension('psfpy.speedy',
               #           sources=['psfpy/fft_speed.c', 'psfpy/speedy.pyx'],
               #           include_dirs=['include', numpy.get_include()],
               #           libraries=['m', 'fftw3'],
               #           library_dirs=['lib'],
               #           extra_compile_args=['-std=c99', '-O3'],
               #           extra_link_args=['-Wl,-rpath,lib']
               #           )
                ]

setup(
    name='psfpy',
    version='0.0.1',
    packages=['psfpy'],
    url='',
    license='MIT',
    author='J. Marcus Hughes',
    author_email='hughes.jmb@gmail.com',
    description='Point spread function modeling and correction',
    ext_modules=cythonize(ext_modules, annotate=True, compiler_directives={'language_level': 3})
)
