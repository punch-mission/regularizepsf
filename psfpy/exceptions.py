
class ValidationError(Exception):
    pass


class ParameterValidationError(ValidationError):
    pass


class InvalidSizeError(ValidationError):
    pass


class EvaluatedModelInconsistentSizeError(ValidationError):
    pass


class ParameterMismatchOnConstructionError(ValidationError):
    pass


class ParameterMismatchOnEvaluationError(ValidationError):
    pass

class UnevaluatedPointError(Exception):
    pass
