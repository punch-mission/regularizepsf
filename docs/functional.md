# Correct using `FunctionalCorrector`
A functional corrector is defined by a set of equations instead of image arrays. 
The functional corrector can be helpful if you know the form of the PSF. You can directly define it. 

```{note}
To correct an image, a `FunctionalCorrector` will be converted to an `ArrayCorrector`. 
If you plan to save a model for many corrections, you may find it more convenient to manually convert to an `ArrayCorrector`
and then save an `ArrayConverter` instead. This is because a `FunctionalCorrector` merely pickles the functions using 
`dill` while `ArrayCorrector` saves using HDF5. For more details, see the [Saving a corrector](save-corrector) section. 
```

## `simple_psf`: the starting point
Every model begins with a `simple_psf`. It requires the first two arguments to be the `x` and `y` coordinates. 
These will often be passed in as arrays so your function should operate in a vectorized manner and be able to output an
array as well. 

```py
from regularizepsf import simple_psf

@simple_psf
def a_basic_example(x, y):
    return x + y
```
You can always evaluate your PSF at a single point to determine its value:
```py
print(a_basic_example(101, 204))
```
Or you can evaluate at a variety of `x` and `y` coordinates using a `numpy` array. 
```py
print(a_basic_example(np.arange(100), np.arange(!00)))
```

You can then supply additional arguments with default values set. We will see in the next section
how to use a `varied_psf` to make them vary across the field-of-view (FOV) of the image. 

```py

@simple_psf
def with_default_arguments(x, y, width=100):
    return x + y + width
```

## `varied_psf`: a more realistic model
The purpose of this package is correct images with variable PSFs. Thus, we need a way to encode how the 
PSF varies across the FOV. That's where `varied_psf` helps. The decorator requires a `simple_psf` as an argument. 
We call this the base PSF. 
Then, the function takes `x` and `y` as parameters and returns a dictionary of how the defaulted parameters of the base PSF vary. 

For example, 
```py
from regularizepsf import simple_psf, varied_psf

@simple_psf
def base_psf(x, y, width=100):
    return x + y + width

@varied_psf(base_psf)
def complicated_psf(x, y):
    return {"width": x/(y+1)}
```

Now, the PSF's width will vary across an image and have the width of `x` divided by `y+1`. (We add one to avoid division
by zero errors.)

## Making a `FunctionalCorrector`
Using these functionally defined examples, we can then create a `FunctionalCorrector` to correct an image. 

```py
from regularizepsf import FunctionalCorrector

my_simple_corrector = FunctionalCorrector(base_psf)
my_complicated_corrector = FunctionalCorrector(complicated_psf)
```

As discussed in the [Quickstart](quickstart.md), we often wish to correct our PSF to a uniform output by applying a 
target PSF. We can provide that too!

```py
@simple_psf
def target_psf(x, y):
    return np.ones_like(x)

my_corrector = FunctionalCorrector(complicated_psf, target_psf)
```

## Correcting an image
Correcting an image is now trivial. We just load the image and correct with a specified patch size, 256 in this case. 
```python
from astropy.io import fits

with fits.open("path_to_image.fits") as hdul:
    data = hdul[0].data.astype(float)

my_corrector.correct_image(data, 256)
```

```{note}
If you're planning to do many corrections, you might first convert to an `ArrayCorrector`. The `FunctionalCorrector`'s
`correct_image` function involves this step and would do it for each image. 
```

You can evaluate to an `ArrayCorrector` as shown below. The first argument is the `x`, then the `y`, and then the `size` of the patches. 
```python
new_corrector = my_corrector.evaluate_to_array_form(np.arange(256), np.arange(256), 256)
```

(save-corrector)=
## Saving a corrector
We can save a corrector in either its `FunctionalCorrector` form or its `ArrayCorrector` form.
```python
my_corrector.save("functional.psf")
new_corrector.save("array.psf")
```