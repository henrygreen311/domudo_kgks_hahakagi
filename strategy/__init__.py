"""
Pluggable trading strategies.

Every module in this package (observation_engine.py, vwap_stg.py,
ema_stg.py, ...) implements the same strategy.base.StrategyEngine
interface, so tracker.py can switch between them by name alone — see
tracker.py's STRATEGY_NAME constant and load_strategy() below.

To add a new strategy: drop a new `strategy/<name>.py` in this folder
following the same shape as the existing ones (a Config dataclass, an
engine class subclassing StrategyEngine, and a `build(ctx)` factory —
see strategy/base.py's module docstring for the exact contract), then
set tracker.py's STRATEGY_NAME to that file's name (without .py).
"""

import importlib

from .base import CandidateLike, StrategyContext, StrategyEngine

__all__ = ["CandidateLike", "StrategyContext", "StrategyEngine", "load_strategy"]


def load_strategy(name: str, ctx: StrategyContext) -> StrategyEngine:
    """Imports strategy/<name>.py and calls its build(ctx). Raises
    ModuleNotFoundError with a clear message if `name` doesn't match any
    file in this package, and AttributeError if that module forgot to
    define build() — both fail loudly at startup rather than silently
    falling back to a default strategy."""
    try:
        module = importlib.import_module(f"strategy.{name}")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"[strategy] no strategy module named '{name}' found in the strategy/ folder "
            f"(expected strategy/{name}.py)"
        ) from exc

    if not hasattr(module, "build"):
        raise AttributeError(f"[strategy] strategy/{name}.py has no build(ctx) factory function")

    engine = module.build(ctx)
    if not isinstance(engine, StrategyEngine):
        raise TypeError(
            f"[strategy] strategy/{name}.py's build() returned {type(engine).__name__}, "
            f"which isn't a strategy.base.StrategyEngine"
        )
    return engine
