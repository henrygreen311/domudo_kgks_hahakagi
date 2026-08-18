"""
Pluggable trading strategies.

Every module in this package implements the same strategy.base.StrategyEngine
interface, so tracker.py can switch between them by name alone — see
tracker.py's STRATEGY_NAME constant, load_strategy(), and
discover_trade_window_ms() below.

To add a new strategy: drop a new `strategy/<name>.py` in this folder
following the same shape as the existing ones (a Config dataclass, an
engine class subclassing StrategyEngine, a `build(ctx)` factory, and
optionally a REQUIRED_TRADE_WINDOW_MS constant — see strategy/base.py's
module docstring for the exact contract), then set tracker.py's
STRATEGY_NAME to that file's name (without .py). Nothing else in
tracker.py needs to change.
"""

import importlib
from typing import Optional

from .base import CandidateLike, StrategyContext, StrategyEngine

__all__ = ["CandidateLike", "StrategyContext", "StrategyEngine", "load_strategy", "discover_trade_window_ms"]


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


def discover_trade_window_ms(name: str) -> Optional[int]:
    """Imports strategy/<name>.py and reads its REQUIRED_TRADE_WINDOW_MS
    module constant, if declared — the largest window_ms it will ever
    pass to trade_store.get_window(). tracker.py calls this BEFORE
    building TradeStore, so retention always matches whichever strategy
    STRATEGY_NAME currently points at, with no per-strategy dict to
    maintain by hand. Returns None if the module can't be imported yet
    (e.g. a bad STRATEGY_NAME — load_strategy() will raise the real
    error later) or simply doesn't declare the constant, in which case
    tracker.py falls back to its base retention window only."""
    try:
        module = importlib.import_module(f"strategy.{name}")
    except ModuleNotFoundError:
        return None
    value = getattr(module, "REQUIRED_TRADE_WINDOW_MS", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
