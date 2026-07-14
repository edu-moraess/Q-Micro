from core.order import Order, Side, OrderType
from core.exchange_simulator import ExchangeSimulator

ex = ExchangeSimulator(symbols=["SYNTH"])

ex.submit_order("SYNTH", Order(side=Side.BUY, price=100.20, quantity=500))
ex.submit_order("SYNTH", Order(side=Side.SELL, price=100.30, quantity=700))
trades = ex.submit_order("SYNTH", Order(side=Side.BUY, price=100.30, quantity=300, order_type=OrderType.LIMIT))

print(ex.market_data("SYNTH"))
print(trades)