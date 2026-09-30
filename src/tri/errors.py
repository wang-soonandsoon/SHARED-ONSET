"""Failures with distinct meanings; zero mass is not a proof of UNSAT."""


class TRIError(Exception):
    """Base class for expected user-facing failures."""


class InvalidSpecification(TRIError, ValueError):
    pass


class UnsupportedSpec(TRIError):
    pass


class BudgetExceeded(TRIError):
    pass


class ZeroMass(TRIError):
    pass


class VerificationError(TRIError):
    pass
