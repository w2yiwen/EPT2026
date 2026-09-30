"""Compatibility CLI facade; use ``python -m eptnet.evaluation.aggregate``."""

from .evaluation import aggregate as _implementation

main = _implementation.main


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
