"""Compatibility CLI facade; use ``python -m eptnet.training.trainer``."""

from .training import trainer as _implementation

main = _implementation.main


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
