"""OEDISI Feed-Forward Neural Network Imputation Federate."""

# Eagerly initialize deep learning runtime to prevent CFFI ABI ordering conflicts with HELICS
try:
    import keras  # noqa: F401
except ImportError:
    pass

__version__ = "0.1.0"
