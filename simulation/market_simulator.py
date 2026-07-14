
"""Q-Micro :: simulation.market_simulator — orchestrates agents + strategies over time."""

from __future__ import annotations
from typing import List, Optional
from data.synthetic_generator import SyntheticMarketGenerator, Regime


class MarketSimulator:
    def __init__(self, symbol: str = "SYNTH", start_price: float = 100.0, seed: int = 42):
        self.generator = SyntheticMarketGenerator(symbol=symbol, start_price=start_price, seed=seed)
        self.history: List[dict] = []

    def configure_regime(self, regime: Regime) -> None:
        self.generator.set_regime(regime)

    def run(self, n_steps: int, regime_schedule: Optional[List[Regime]] = None) -> List[dict]:
        for step in range(n_steps):
            if regime_schedule and step < len(regime_schedule):
                self.generator.set_regime(regime_schedule[step])
            snapshot = self.generator.step()
            snapshot["step"] = step
            self.history.append(snapshot)
        return self.history

    def trade_tape(self) -> List[dict]:
        return self.generator.ex.trade_tape(self.generator.symbol)