"""
Q-Micro :: dashboard.streamlit_app
--------------------------------------
Terminal de pesquisa em microestrutura de mercado.
Rodar com: streamlit run dashboard/streamlit_app.py
"""

import sys
import os
import asyncio
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ----------------------------------------------------------------------
# Módulo realtime (opcional)
try:
    from realtime.websocket_client import BinanceWebSocketClient
    from realtime.market_stream import MarketStream
    from realtime.event_dispatcher import EventDispatcher
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False

# ----------------------------------------------------------------------
# Módulo replay (opcional)
try:
    from replay.playback_controller import PlaybackController
    REPLAY_AVAILABLE = True
except ImportError:
    REPLAY_AVAILABLE = False

# ----------------------------------------------------------------------
# Monitor de performance
try:
    from performance.monitor import PerformanceMonitor
    if "perf_monitor" not in st.session_state:
        st.session_state.perf_monitor = PerformanceMonitor()
    PERF_AVAILABLE = True
except ImportError:
    PERF_AVAILABLE = False
    if "perf_monitor" not in st.session_state:
        st.session_state.perf_monitor = None

# ----------------------------------------------------------------------
# Imports do Q-Micro – módulos principais
from core.exchange_simulator import ExchangeSimulator
from core.order import Side, OrderType
from data.synthetic_generator import SyntheticMarketGenerator

# ----------------------------------------------------------------------
# Módulos de microestrutura (importação condicional)
try:
    from microstructure.kyle_lambda import KyleLambda
    KYLE_AVAILABLE = True
except ImportError:
    KYLE_AVAILABLE = False

try:
    from microstructure.vpin import VPIN
    VPIN_AVAILABLE = True
except ImportError:
    VPIN_AVAILABLE = False

# ----------------------------------------------------------------------
# Algoritmos de execução (importação condicional)
try:
    from execution.twap import TWAP
    from execution.vwap import VWAP
    from execution.implementation_shortfall import ImplementationShortfall
    from execution.optimal_execution import OptimalExecution
    EXECUTION_AVAILABLE = True
except ImportError:
    EXECUTION_AVAILABLE = False

# ----------------------------------------------------------------------
st.set_page_config(page_title="Q-Micro Terminal", layout="wide")

# ----------------------------------------------------------------------
# Inicializa simulador sintético
if "exchange" not in st.session_state:
    try:
        st.session_state.exchange = ExchangeSimulator()
    except Exception as e:
        st.error(f"Falha ao inicializar ExchangeSimulator: {e}")
        st.stop()
    st.session_state.synthetic_generator = SyntheticMarketGenerator()
    st.session_state.trade_history = []
    st.session_state.order_book_history = []

# ----------------------------------------------------------------------
# Inicializa live stream
if "live_initialized" not in st.session_state:
    st.session_state.live_initialized = False
    if REALTIME_AVAILABLE:
        st.session_state.live_dispatcher = EventDispatcher()
        st.session_state.live_stream = MarketStream("btcusdt", st.session_state.live_dispatcher)
        st.session_state.live_orderbook = {"bid": None, "ask": None, "bid_qty": 0, "ask_qty": 0}
        st.session_state.live_trades = []

        def on_orderbook(data):
            st.session_state.live_orderbook = data
        def on_trade(data):
            st.session_state.live_trades.append(data)
            if len(st.session_state.live_trades) > 100:
                st.session_state.live_trades.pop(0)

        st.session_state.live_dispatcher.subscribe("orderbook", on_orderbook)
        st.session_state.live_dispatcher.subscribe("trade", on_trade)

        def start_ws():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            client = BinanceWebSocketClient(
                ["btcusdt@depth@100ms", "btcusdt@trade"],
                st.session_state.live_stream.handle_message
            )
            loop.run_until_complete(client.connect())
        threading.Thread(target=start_ws, daemon=True).start()
    st.session_state.live_initialized = True

# ----------------------------------------------------------------------
# Inicializa replay engine
if "playback_ctrl" not in st.session_state and REPLAY_AVAILABLE:
    st.session_state.playback_ctrl = PlaybackController(data_dir="data/recorded")
    st.session_state.replay_engine = None
    st.session_state.replay_playing = False

# ----------------------------------------------------------------------
# Função auxiliar para obter estado do book de forma segura
def get_order_book_state(exchange):
    """Retorna um dicionário com best_bid, best_ask, mid_price, spread, buy_depth, sell_depth."""
    try:
        # Tenta o método original
        if hasattr(exchange, 'get_order_book_state'):
            return exchange.get_order_book_state()
        # Fallback: constrói a partir dos atributos acessíveis
        ob = exchange.order_book
        best_bid = max(ob.bids.keys()) if ob.bids else None
        best_ask = min(ob.asks.keys()) if ob.asks else None
        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else None
        spread = (best_ask - best_bid) if best_bid and best_ask else None
        buy_depth = [(price, qty) for price, qty in sorted(ob.bids.items(), reverse=True)[:5]] if ob.bids else []
        sell_depth = [(price, qty) for price, qty in sorted(ob.asks.items())[:5]] if ob.asks else []
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid_price': mid,
            'spread': spread,
            'buy_depth': buy_depth,
            'sell_depth': sell_depth
        }
    except Exception as e:
        st.warning(f"Não foi possível obter o estado do livro de ofertas: {e}")
        return {
            'best_bid': None, 'best_ask': None, 'mid_price': None, 'spread': None,
            'buy_depth': [], 'sell_depth': []
        }

# ----------------------------------------------------------------------
# Sidebar
with st.sidebar:
    st.title("🎛️ Q-Micro Controls")

    st.header("Simulation")
    if st.button("🔄 Reset Simulation"):
        try:
            st.session_state.exchange = ExchangeSimulator()
        except Exception as e:
            st.error(f"Erro ao resetar simulação: {e}")
        st.session_state.trade_history = []
        st.session_state.order_book_history = []
        st.rerun()

    st.header("Market Parameters")
    n_traders = st.slider("Number of Traders", 1, 50, 10)
    initial_price = st.number_input("Initial Price", value=100.0, step=0.1)
    volatility = st.slider("Volatility", 0.001, 0.1, 0.02, step=0.001)

    st.header("Execution Parameters")
    execution_algorithm = st.selectbox(
        "Algorithm",
        ["TWAP", "VWAP", "Implementation Shortfall", "Optimal Execution"],
    )
    total_quantity = st.number_input("Total Quantity", value=1000, step=100)
    n_slices = st.slider("Number of Slices", 1, 50, 10)

    st.header("RL Agent")
    use_rl_agent = st.checkbox("Enable RL Agent")
    if use_rl_agent:
        rl_algorithm = st.selectbox("RL Algorithm", ["DQN", "PPO"])

# ----------------------------------------------------------------------
st.title("📊 Q-Micro Dashboard")
tabs = st.tabs([
    "📡 Live Order Book",
    "📈 Order Book",
    "💹 Market Impact",
    "🚀 Execution Analysis",
    "⚙️ Simulation Control",
    "⏪ Replay",
    "📊 Performance"
])

# ======================================================================
# Aba 0: Live Order Book
# ======================================================================
with tabs[0]:
    st.header("📡 BTC/USDT – Livro de Ofertas em Tempo Real")
    if not REALTIME_AVAILABLE:
        st.warning("Módulo 'realtime' não encontrado. A funcionalidade live está desativada.")
    else:
        if st.button("🔄 Atualizar dados"):
            pass
        lob = st.session_state.live_orderbook
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Melhor Bid", f"{lob['bid']:.2f}" if lob['bid'] else "N/A")
        col2.metric("Qtd Bid", f"{lob['bid_qty']:.4f}" if lob['bid_qty'] else "N/A")
        col3.metric("Melhor Ask", f"{lob['ask']:.2f}" if lob['ask'] else "N/A")
        col4.metric("Qtd Ask", f"{lob['ask_qty']:.4f}" if lob['ask_qty'] else "N/A")
        st.write("---")
        st.write("**Últimos trades recebidos:**")
        if st.session_state.live_trades:
            trades_df = pd.DataFrame(st.session_state.live_trades)
            st.dataframe(trades_df.tail(10), use_container_width=True)
        else:
            st.info("Nenhum trade recebido ainda. Aguardando conexão...")

# ======================================================================
# Aba 1: Order Book (sintético)
# ======================================================================
with tabs[1]:
    st.header("Order Book Visualization")
    ob_state = get_order_book_state(st.session_state.exchange)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Bid", f"{ob_state['best_bid']:.2f}" if ob_state['best_bid'] else "N/A")
    col2.metric("Best Ask", f"{ob_state['best_ask']:.2f}" if ob_state['best_ask'] else "N/A")
    col3.metric("Mid Price", f"{ob_state['mid_price']:.2f}" if ob_state['mid_price'] else "N/A")
    col4.metric("Spread", f"{ob_state['spread']:.4f}" if ob_state['spread'] else "N/A")

    if ob_state['buy_depth'] or ob_state['sell_depth']:
        fig = go.Figure()
        if ob_state['buy_depth']:
            buy_prices, buy_quantities = zip(*ob_state['buy_depth'])
            fig.add_trace(go.Bar(x=list(buy_prices), y=list(buy_quantities), name="Bid", marker_color="green"))
        if ob_state['sell_depth']:
            sell_prices, sell_quantities = zip(*ob_state['sell_depth'])
            fig.add_trace(go.Bar(x=list(sell_prices), y=list(sell_quantities), name="Ask", marker_color="red"))
        fig.update_layout(title="Order Book Depth", barmode="group", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Raw Order Book Data"):
        st.write("### Buy Side (BID)")
        if ob_state['buy_depth']:
            st.dataframe(pd.DataFrame(ob_state['buy_depth'], columns=["Price", "Quantity"]))
        st.write("### Sell Side (ASK)")
        if ob_state['sell_depth']:
            st.dataframe(pd.DataFrame(ob_state['sell_depth'], columns=["Price", "Quantity"]))

# ======================================================================
# Aba 2: Market Impact
# ======================================================================
with tabs[2]:
    st.header("Market Impact Analysis")

    if not KYLE_AVAILABLE or not VPIN_AVAILABLE:
        st.warning("Algumas métricas de microestrutura não estão disponíveis (módulos ausentes).")

    # Tenta acessar a lista de trades
    trades_list = []
    try:
        trades_list = st.session_state.exchange.order_book.trades
    except AttributeError:
        pass

    if len(trades_list) > 0:
        trades_df = pd.DataFrame(trades_list)
        buy_volume = trades_df[trades_df["buyer"] == "RL_AGENT"]["quantity"].sum()
        sell_volume = trades_df[trades_df["seller"] == "RL_AGENT"]["quantity"].sum()
        ofi = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-6)

        vpin_value = None
        if VPIN_AVAILABLE:
            vpin = VPIN(bucket_size=10)
            vpin_value = vpin.compute_vpin_from_trades(trades_list)

        kyle_impact = None
        if KYLE_AVAILABLE:
            kyle = KyleLambda(lambda_=0.01)
            kyle_impact = kyle.estimate_impact(ofi)

        col1, col2, col3 = st.columns(3)
        col1.metric("Order Flow Imbalance", f"{ofi:.4f}")
        col2.metric("VPIN", f"{vpin_value:.4f}" if vpin_value is not None else "N/A")
        col3.metric("Kyle's Lambda Impact", f"{kyle_impact:.4f}" if kyle_impact is not None else "N/A")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trades_df.index, y=trades_df["price"], mode="lines+markers", name="Trade Price"))
        fig.update_layout(title="Trade Price History", height=400)
        st.plotly_chart(fig, use_container_width=True)

        trades_df["cumulative_volume"] = trades_df["quantity"].cumsum()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=trades_df.index, y=trades_df["cumulative_volume"], mode="lines", name="Cumulative Volume"))
        fig2.update_layout(title="Cumulative Trade Volume", height=400)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No trades executed yet. Run a simulation to see market impact metrics.")

# ======================================================================
# Aba 3: Execution Analysis
# ======================================================================
with tabs[3]:
    st.header("Execution Algorithm Analysis")
    if not EXECUTION_AVAILABLE:
        st.warning("Algoritmos de execução não disponíveis (módulos ausentes).")
    else:
        if st.button("🚀 Run Execution Algorithm"):
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=30)

            # Preço de decisão: tentar obter mid_price do estado atual
            ob_state = get_order_book_state(st.session_state.exchange)
            decision_price = ob_state['mid_price'] or 100.0

            if execution_algorithm == "TWAP":
                twap = TWAP(total_quantity=total_quantity, start_time=start_time, end_time=end_time, n_slices=n_slices)
                trades = twap.execute(st.session_state.exchange, side="BUY")
                st.session_state.trade_history = trades
            elif execution_algorithm == "VWAP":
                volume_profile = [100 + i * 10 for i in range(n_slices)]
                vwap = VWAP(total_quantity=total_quantity, start_time=start_time, end_time=end_time, volume_profile=volume_profile)
                trades = vwap.execute(st.session_state.exchange, side="BUY")
                st.session_state.trade_history = trades
            elif execution_algorithm == "Implementation Shortfall":
                is_strategy = ImplementationShortfall(
                    total_quantity=total_quantity, start_time=start_time, end_time=end_time,
                    decision_price=decision_price,
                    lambda_=0.01, sigma=volatility, risk_aversion=0.5, n_slices=n_slices
                )
                trades = is_strategy.execute(st.session_state.exchange, side="BUY")
                st.session_state.trade_history = trades
            elif execution_algorithm == "Optimal Execution":
                oe_strategy = OptimalExecution(
                    total_quantity=total_quantity, start_time=start_time, end_time=end_time,
                    decision_price=decision_price,
                    sigma=volatility, lambda_=0.01, eta=0.005, risk_aversion=0.5, n_slices=n_slices
                )
                trades = oe_strategy.execute(st.session_state.exchange, side="BUY")
                st.session_state.trade_history = trades

            st.success(f"✅ Executed {len(trades)} trades using {execution_algorithm}!")

        if st.session_state.trade_history:
            trades_df = pd.DataFrame(st.session_state.trade_history)
            st.write("### Trade History")
            st.dataframe(trades_df, use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trades_df.index, y=trades_df["price"], mode="lines+markers", name="Execution Price"))
            fig.update_layout(title=f"{execution_algorithm} Execution Prices", height=400)
            st.plotly_chart(fig, use_container_width=True)

            avg_price = trades_df["price"].mean()
            total_volume = trades_df["quantity"].sum()
            price_std = trades_df["price"].std()
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Execution Price", f"{avg_price:.2f}")
            col2.metric("Total Volume Executed", f"{total_volume:,}")
            col3.metric("Price Volatility", f"{price_std:.4f}")

# ======================================================================
# Aba 4: Simulation Control
# ======================================================================
with tabs[4]:
    st.header("Simulation Control")
    if st.button("🎲 Generate Synthetic Orders"):
        n_orders = st.slider("Number of Orders", 10, 1000, 100, key="n_orders")
        st.session_state.synthetic_generator = SyntheticMarketGenerator(n_traders=n_traders, initial_price=initial_price)
        orders = st.session_state.synthetic_generator.generate_order_flow(n_orders)
        for order in orders:
            side = Side.BUY if order["side"] == "BUY" else Side.SELL
            st.session_state.exchange.submit_order(
                trader_id=order["trader_id"], side=side,
                price=order["price"], quantity=order["quantity"], order_type=OrderType.LIMIT
            )
        st.success(f"✅ Generated and submitted {n_orders} synthetic orders!")

    if use_rl_agent and st.button("🤖 Run RL Agent"):
        from strategies.rl_environment import TradingEnvironment
        from strategies.rl_execution_agent import RLExecutionAgent
        env = TradingEnvironment(order_book=st.session_state.exchange.order_book, max_steps=100, target_quantity=100)
        agent = RLExecutionAgent(algorithm=rl_algorithm)
        st.info(f"Training {rl_algorithm} agent for 5 episodes...")
        episode_rewards = agent.train(env, episodes=5, render=False)
        st.success(f"✅ RL Agent trained! Episode rewards: {episode_rewards}")
        agent.save(f"{rl_algorithm}_agent.pth")
        st.info(f"Model saved as {rl_algorithm}_agent.pth")

    if st.session_state.order_book_history:
        st.write("### Order Book History")
        st.dataframe(pd.DataFrame(st.session_state.order_book_history), use_container_width=True)

# ======================================================================
# Aba 5: Replay
# ======================================================================
with tabs[5]:
    st.header("⏪ Replay de Sessões Históricas")
    if not REPLAY_AVAILABLE:
        st.warning("Módulo 'replay' não disponível.")
    else:
        sessions = st.session_state.playback_ctrl.list_sessions()
        if sessions:
            selected_session = st.selectbox("Selecionar sessão", sessions)
            col1, col2, col3 = st.columns(3)
            if col1.button("Carregar sessão"):
                with st.spinner("Carregando..."):
                    engine = st.session_state.playback_ctrl.load_session(selected_session)
                    st.session_state.replay_engine = engine
                    st.success("Sessão carregada!")
            if col2.button("Play"):
                if st.session_state.replay_engine:
                    st.session_state.replay_engine.play()
                    def run():
                        st.session_state.replay_engine.run_playback_loop()
                    threading.Thread(target=run, daemon=True).start()
            if col3.button("Pause"):
                if st.session_state.replay_engine:
                    st.session_state.replay_engine.pause()

            speed = st.slider("Velocidade", 0.1, 10.0, 1.0, 0.1)
            if st.session_state.replay_engine:
                st.session_state.replay_engine.set_speed(speed)

            if st.session_state.replay_engine and hasattr(st.session_state.replay_engine, 'trades'):
                end_idx = st.session_state.replay_engine.current_index
                replay_df = st.session_state.replay_engine.trades.iloc[max(0,end_idx-10):end_idx]
                st.dataframe(replay_df)
        else:
            st.info("Nenhuma sessão gravada encontrada. Use o recorder para salvar dados primeiro.")

# ======================================================================
# Aba 6: Performance
# ======================================================================
with tabs[6]:
    st.header("📊 Monitor de Performance")
    if not PERF_AVAILABLE:
        st.warning("Módulo 'performance' não disponível (psutil não instalado).")
    else:
        perf = st.session_state.perf_monitor.snapshot()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Eventos/s", f"{perf['events_per_sec']:.1f}")
        col2.metric("Latência média", f"{perf['avg_latency_ms']:.1f} ms")
        col3.metric("CPU", f"{perf['cpu_percent']:.1f}%")
        col4.metric("Memória", f"{perf['memory_percent']:.1f}%")
        st.write("**Contagem de eventos:**")
        st.json(perf['event_counts'])