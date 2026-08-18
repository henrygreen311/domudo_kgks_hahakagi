"""
Common interface every strategy module implements, so tracker.py can
swap between them purely by name — see tracker.py's STRATEGY_NAME —
without any other code in tracker.py needing to change, even across
hundreds of different strategy files being tried over time.

A strategy module (a .py file living in this `strategy/` package) must
expose:

  - A config dataclass, all fields defaulted — every field needs a
    sensible default since StrategyContext.build_config() may construct
    it with zero kwargs.

  - An engine class subclassing StrategyEngine below, implementing
    sync_watchlist / evaluate / snapshot.

  - A free function `build(ctx: StrategyContext) -> StrategyEngine` that
    tracker.py's main() calls once at startup. It's a free function
    rather than a fixed constructor signature because different
    strategies need different inputs — e.g. one strategy might only read
    candles and never touch ctx.trade_store, while another needs
    trade_store but never touches ctx.candle_fetcher. Each strategy's
    build() just takes what it needs off ctx and ignores the rest.

  - OPTIONALLY, a module-level `REQUIRED_TRADE_WINDOW_MS: int` constant —
    the single largest window_ms value this strategy will ever pass to
    trade_store.get_window(). tracker.py reads this automatically via
    strategy.discover_trade_window_ms(STRATEGY_NAME) *before* building
    TradeStore, so TradeStore always retains enough history for whichever
    strategy is currently selected — no per-strategy dict to maintain in
    tracker.py, and no silent data-starvation if a strategy needs more
    than the base 5-second retention. Omit this constant entirely if the
    strategy never touches ctx.trade_store at all.

Nothing here enforces the internal shape of a "candidate" — only that
whatever evaluate() decides is ready gets turned into a market_data.Signal
before being returned, and that whatever snapshot() returns has a
`.symbol` and a `.status_line()` for tracker.py's generic periodic
status log.
"""

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from market_data import MarketDataStore, TradeStore, Signal

log = logging.getLogger("okx_futures.strategy")


@runtime_checkable
class CandidateLike(Protocol):
    symbol: str

    def status_line(self) -> str: ...


@dataclass
class StrategyContext:
    """Everything tracker.py's main() can hand to any strategy's build()
    factory. Individual strategies only read what they actually need —
    see this class's own docstring above for why the signature is this
    permissive rather than one constructor per strategy."""

    market_data: MarketDataStore
    trade_store: Optional[TradeStore] = None
    candle_fetcher: Optional[Any] = None  # Callable[[symbol, bar, limit], Awaitable[List[dict]]] — see okx_futures_client.get_candles
    okx_client: Optional[Any] = None
    # Plain dict of config-field overrides, keyed by the strategy's own
    # config dataclass field names — see tracker.py's STRATEGY_OVERRIDES.
    # Kept as a dict rather than a config instance so tracker.py never
    # needs to import any individual strategy's config class.
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    def build_config(self, config_cls):
        """Instantiates `config_cls` with self.config_overrides applied
        on top of its own defaults. Unknown override keys (e.g. leftover
        tuning meant for a different strategy) are dropped with a
        warning rather than raising, so switching STRATEGY_NAME in
        tracker.py never crashes startup over a stale override."""
        overrides = dict(self.config_overrides or {})
        try:
            return config_cls(**overrides)
        except TypeError:
            valid = {f for f in getattr(config_cls, "__dataclass_fields__", {})}
            dropped = {k: v for k, v in overrides.items() if k not in valid}
            kept = {k: v for k, v in overrides.items() if k in valid}
            if dropped:
                log.warning(
                    f"[strategy] {config_cls.__name__} ignoring override key(s) it doesn't have "
                    f"a field for: {sorted(dropped)}"
                )
            return config_cls(**kept)


class StrategyEngine(abc.ABC):
    """Base class for a strategy's live engine (e.g.
    observation_engine.ObservationWindowManager). tracker.py's
    run_trading_loop talks only to this interface, never to any
    strategy-specific internals — that's what makes swapping strategies
    a one-line change in tracker.py instead of a rewrite."""

    #: Short, human-readable name used in startup/status logging so it's
    #: obvious from the logs which strategy is actually running.
    name: str = "strategy"

    @abc.abstractmethod
    async def sync_watchlist(self, watchlist_symbols) -> None:
        """Start tracking any symbol newly present in the watchlist and
        drop local state for any symbol that fell off it."""
        raise NotImplementedError

    @abc.abstractmethod
    async def evaluate(self, symbol: str) -> Optional[Signal]:
        """Run one fresh check for `symbol`. Returns a ready-to-open
        market_data.Signal the instant this strategy's own entry
        conditions are met, else None — including on every tick where
        the read simply doesn't qualify yet, which is not an error."""
        raise NotImplementedError

    @abc.abstractmethod
    async def snapshot(self) -> List[CandidateLike]:
        """Current in-flight candidates, for tracker.py's periodic
        condensed status log (see tracker.py's run_trading_loop)."""
        raise NotImplementedError
