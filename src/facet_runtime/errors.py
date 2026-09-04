"""Errors exposed by the narrow Facet runtime contract."""


class FacetRuntimeError(RuntimeError):
    """Base class for expected runtime failures."""


class BackendUnavailableError(FacetRuntimeError):
    """Raised when a requested compute backend is unavailable."""


class BackendMismatchError(FacetRuntimeError):
    """Raised when a runtime did not use the requested compute backend."""
