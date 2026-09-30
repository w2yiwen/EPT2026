"""Compatibility CLI facade; use ``python -m eptnet.evaluation.evaluator``."""

from .evaluation import evaluator as _implementation

main = _implementation.main


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
