# Q-Micro: Real-Time Market Microstructure & Execution Research Engine

> **Status:** Em desenvolvimento
> **Autor:** Eduardo Moraes
> **Foco:** Simulação de microestrutura de mercado para pesquisa em HFT e execução ótima.

---

## 📌 Descrição
O **Q-Micro** é um **simulador de microestrutura de mercado** projetado para pesquisa em **High-Frequency Trading (HFT)** e **execução ótima de ordens**. O projeto implementa um **Limit Order Book (LOB)**, **Matching Engine**, **modelos de microestrutura**, **algoritmos de execução institucional** e um **agente de Reinforcement Learning (RL)** para otimização de estratégias.

---

## 🛠️ Estrutura do Projeto
```text
Q-Micro/
├── data/                  # Loaders e geradores de dados
├── core/                  # Núcleo: Order, OrderBook, MatchingEngine, ExchangeSimulator
├── microstructure/        # Modelos: Spread, Liquidez, Impacto, VPIN, Kyle Lambda
├── execution/             # Algoritmos: TWAP, VWAP, Implementation Shortfall
├── strategies/            # Estratégias: Market Maker, Liquidity Provider
├── simulation/            # Simuladores: Market Simulator, Monte Carlo
├── analytics/             # Métricas: Performance, Risco
├── dashboard/             # Visualização: Streamlit App
├── tests/                 # Testes unitários
├── requirements.txt       # Dependências
└── README.md              # Documentação

Como Começar
1. Instalar dependências
pip install -r requirements.txt
2. Executar um teste básico
from core.exchange_simulator import ExchangeSimulator
from core.order import OrderSide, OrderType

exchange = ExchangeSimulator()
exchange.submit_order("TRADER_1", OrderSide.BUY, 100.0, 500, OrderType.LIMIT)
exchange.submit_order("TRADER_2", OrderSide.SELL, 100.5, 300, OrderType.LIMIT)
exchange.submit_order("TRADER_3", OrderSide.BUY, 0.0, 200, OrderType.MARKET)

print(exchange.get_order_book_state())
3. Iniciar o dashboard (Streamlit)
streamlit run dashboard/streamlit_app.py
📚 Documentação
- Arquitetura do Projeto (em breve)
- Modelos de Microestrutura (em breve)
- Algoritmos de Execução (em breve)
🤝 Contribuições
Contribuições são bem-vindas! Abra um Pull Request ou report um Issue.
📜 Licença
MIT

---

---

### **📁 `data/`**
---

#### **1. `data/__init__.py`**
```python
"""
Data module for Q-Micro.
Includes market data loaders and synthetic data generators.
"""