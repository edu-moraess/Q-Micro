# Q-Micro: Real-Time Market Microstructure & Execution Research Engine

Research infrastructure to study price formation, liquidity, order impact,
execution costs, order-book dynamics, and institutional execution algorithms —
built as a self-contained matching-engine simulator, not a price predictor.

## Status

| Phase | Module | Status |
|---|---|---|
| 1-2 | `core/order.py`, `core/order_book.py` | ✅ Done |
| 3 | `core/matching_engine.py`, `core/exchange_simulator.py` | ✅ Done |
| 4 | `data/synthetic_generator.py` (noise / informed / MM / institutional agents) | ⏳ Pending |
| 5 | `microstructure/` (spread, Kyle λ, Amihud, VPIN, OFI) | ⏳ Pending |
| 6 | `execution/` (TWAP, VWAP, Implementation Shortfall) | ⏳ Pending |
| 7 | RL execution agent (PyTorch, DQN/PPO) | ⏳ Pending |
| 8 | Performance optimization (Numba/Polars) | ⏳ Pending |
| 9 | `dashboard/streamlit_app.py` | ⏳ Pending |
| 10 | Research report | ⏳ Pending |

## Architecture

Price-time priority matching engine with lazy-deleted heap-based order book
(O(log n) best-bid/ask, O(1) amortized top-of-book reads). No external
dependencies for the core engine — pure stdlib (`heapq`, `dataclasses`,
`enum`, `collections`).

## Quickstart

\`\`\`python
from core.order import Order, Side, OrderType
from core.exchange_simulator import ExchangeSimulator

ex = ExchangeSimulator(symbols=["SYNTH"])
ex.submit_order("SYNTH", Order(side=Side.BUY, price=100.20, quantity=500))
ex.submit_order("SYNTH", Order(side=Side.SELL, price=100.30, quantity=700))
trades = ex.submit_order("SYNTH", Order(side=Side.BUY, price=100.30, quantity=300))

print(ex.market_data("SYNTH"))
\`\`\`

## Limitations

Research/educational simulator. No real market data feed, no latency
modeling of physical network/exchange colocation, simplified STOP-order
triggering (last-trade based, not full book-sweep).