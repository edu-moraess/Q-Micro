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