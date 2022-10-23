import inspect
from functools import partial, update_wrapper, wraps


class Equation:
    def __init__(self, func):
        update_wrapper(self, func)
        self.func = func
        self.parameters = set(inspect.signature(self.func).parameters.keys())

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def _equation(func):
    return Equation(func)


def equation(number=None):
    if callable(number):
        return _equation(number) # return 'wrapper'
    else:
        raise Exception("called with arguments!")


@equation
def eq(x, y):
    return x + y


class Parameterization:
    def __init__(self, variable_function, reference_function, check_at_call=True):
        self.variable_function = variable_function
        # print(self.variable_function.parameters)
        self.reference_function = reference_function
        print(self.reference_function._parameters)
        self.check_at_call = True

    def __call__(self, *args, **kwargs):
        if self.check_at_call:
            print("CHECKED!")
        return self.variable_function(*args, *kwargs)


# def parameterization(__fn=None, *, reference_function=None, check_at_call=True):
#     if reference_function is None:
#         raise Exception("ERROR!")
#
#     if __fn:
#         return Parameterization(__fn, reference_function, check_at_call=check_at_call)
#     else:
#         return partial(parameterization, reference_function=reference_function, check_at_call=check_at_call)

def _parameterization(reference):
    if reference is None:
        raise Exception("ERROR!")
    print("hi")

    # @wraps(reference)
    def inner(__fn=None, *, check_at_call=True):
        print(__fn is None, reference is None)
        print(__fn, reference)
        if __fn:
            print("there")
            return Parameterization(__fn, reference, check_at_call=check_at_call)
        else:
            print('here')
            return partial(inner, check_at_call=check_at_call)
    return inner


def parameterization(number=None):
    if isinstance(number, Equation):
        return _parameterization(number) # return 'wrapper'
    else:
        if callable(number):
            raise Exception("blank?!") # ... or 'decorator'
        else:
            raise Exception("number")


@parameterization(eq)
def what(x, y):
    return {"sigma": 5}


print(type(what))
#print(what(5, 10))
# print(eq)
# print(what(5, 5))
