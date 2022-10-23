
class ValidationError(Exception):
    pass


class PSFParameterValidationError(ValidationError):
    pass


class InvalidSizeError(ValidationError):
    pass


class EvaluatedModelInconsistentSizeError(ValidationError):
    pass


class VariedPSFParameterMismatchError(ValidationError):
    pass


class UnevaluatedPointError(Exception):
    pass
