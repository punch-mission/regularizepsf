
class ValidationError(Exception):
    pass


class ParameterValidationError(ValidationError):
    pass


class InvalidSizeError(ValidationError):
    pass


class EvaluatedModelInconsistentSizeError(ValidationError):
    pass


class UnevaluatedPointError(Exception):
    pass
