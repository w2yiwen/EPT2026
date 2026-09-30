"""Compatibility facade; use :mod:`eptnet.training.losses` in new code."""

from .training import losses as _implementation

EPTNetLoss = _implementation.EPTNetLoss


def __getattr__(name: str):
    return getattr(_implementation, name)
