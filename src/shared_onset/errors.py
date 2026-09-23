"""Failures with distinct meanings; zero mass is not a proof of UNSAT."""


class SharedOnsetError(Exception):
    """Base class for expected user-facing failures."""


class InvalidSpecification(SharedOnsetError, ValueError):
    pass


class UnsupportedSpec(SharedOnsetError):
    pass


class BudgetExceeded(SharedOnsetError):
    pass


class ZeroMass(SharedOnsetError):
    pass


class VerificationError(SharedOnsetError):
    pass
