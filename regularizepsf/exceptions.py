"""Errors and warnings for regularizepsf."""


class RegularizePSFError(Exception):
    """Base class for regularizepsf exceptions."""


class InvalidCoordinateError(RegularizePSFError):
    """The key for this coordinate does not exist in the model."""


class IncorrectShapeError(RegularizePSFError):
    """The shapes do not match for the model and the value."""


class InvalidFunctionError(RegularizePSFError):
    """Function for functional model has invalid parameters."""


class FunctionParameterMismatchError(RegularizePSFError):
    """Function evaluated with nonexistent kwargs."""

class PSFBuilderError(RegularizePSFError):
    """Something went wrong building the PSF model."""
