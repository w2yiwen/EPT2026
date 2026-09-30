"""Compatibility facade for the pre-0.2 training engine import path."""

from .training import loop as _implementation

Trainer = _implementation.Trainer
move_to_device = _implementation.move_to_device


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_implementation)))
