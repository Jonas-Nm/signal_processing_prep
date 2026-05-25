"""Package-specific errors used to distinguish policy from data failures."""


class SignalProcessingError(ValueError):
    """Base error for invalid signal-processing workflow inputs."""


class ConfigurationError(SignalProcessingError):
    """Raised when analysis policy or configuration is invalid."""


class RecordDataError(SignalProcessingError):
    """Raised when one input record cannot be processed as configured."""
