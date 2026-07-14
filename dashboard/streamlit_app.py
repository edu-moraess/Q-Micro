"""
Streamlit Dashboard for Q-Micro.
Visualizes Order Book, Market Impact, Execution Algorithms, and Simulation Controls.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import sys
from pathlib import Path

# Ajuste para encontrar os módulos na raiz do repositório
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import Q-Micro modules
from core.exchange_simulator import ExchangeSimulator
from core.order import OrderSide, OrderType
from microstructure.spread_model import SpreadModel
from microstructure.liquidity import OrderFlowImbalance, AmihudIlliquidity
from microstructure.kyle_lambda import KyleLambda
from microstructure.vpin import VPIN
from execution.twap import TWAP
from execution.vwap import VWAP
from execution.implementation_shortfall import ImplementationShortfall
from execution.optimal_execution import OptimalExecution
from data.synthetic_generator import SyntheticMarketGenerator

# Page configuration
st.set_page_config(
    page_title="Q-Micro Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "exchange" not in st.session_state:
    st.session_state.exchange = ExchangeSimulator()
    st.session_state.synthetic_generator = SyntheticMarketGenerator()
    st.session_state.simulation_running = False
    st.session_state.trade_history = []
    st.session_state.order_book_history = []

# Sidebar
with st.sidebar:
    st.title("🎛️ Q-Micro Controls")

    # Simulation controls
    st.header("Simulation")
    if st.button("🔄 Reset Simulation"):
        st.session_state.exchange = ExchangeSimulator()
        st.session_state.trade_history = []
        st.session_state.order_book_history = []
        st.rerun()

    # Market parameters
    st.header("Market Parameters")
    n_traders = st.slider("Number of Traders", 1, 50, 10)
    initial_price = st.number_input("Initial Price", value=100.0, step=0.1)
    volatility = st.slider("Volatility", 0.001, 0.1, 0.02, step=0.001)

    # Execution parameters
    st.header("Execution Parameters")
    execution_algorithm = st.selectbox(
        "Algorithm",
        ["TWAP", "VWAP", "Implementation Shortfall", "Optimal Execution"],
    )
    total_quantity = st.number_input("Total Quantity", value=1000, step=100)
    n_slices = st.slider("Number of Slices", 1, 50, 10)

    # RL Agent
    st.header("RL Agent")
    use_rl_agent = st.checkbox("Enable RL Agent")
    if use_rl_agent:
        rl_algorithm = st.selectbox("RL Algorithm", ["DQN", "PPO"])

# Main dashboard
st.title("📊 Q-Micro Dashboard")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Order Book",
    "💹 Market Impact",
    "🚀 Execution Analysis",
    "⚙️ Simulation Control",
])

# Tab 1: Order Book
with tab1:
    st.header("Order Book Visualization")

    # Get current order book state
    ob_state = st.session_state.exchange.get_order_book_state()

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Bid", f"{ob_state['best_bid']:.2f}" if ob_state['best_bid'] else "N/A")
    with col2:
        st.metric("Best Ask", f"{ob_state['best_ask']:.2f}" if ob_state['best_ask'] else "N/A")
    with col3:
        st.metric("Mid Price", f"{ob_state['mid_price']:.2f}" if ob_state['mid_price'] else "N/A")
    with col4:
        st.metric("Spread", f"{ob_state['spread']:.4f}" if ob_state['spread'] else "N/A")

    # Plot order book depth
    if ob_state['buy_depth'] or ob_state['sell_depth']:
        fig = go.Figure()

        # Buy side (BID)
        if ob_state['buy_depth']:
            buy_prices, buy_quantities = zip(*ob_state['buy_depth'])
            fig.add_trace(go.Bar(
                x=list(buy_prices),
                y=list(buy_quantities),
                name="Bid",
                marker_color="green",
                orientation="v",
            ))

        # Sell side (ASK)
        if ob_state['sell_depth']:
            sell_prices, sell_quantities = zip(*ob_state['sell_depth'])
            fig.add_trace(go.Bar(
                x=list(sell_prices),
                y=list(sell_quantities),
                name="Ask",
                marker_color="red",
                orientation="v",
            ))

        fig.update_layout(
            title="Order Book Depth",
            xaxis_title="Price",
            yaxis_title="Quantity",
            barmode="group",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Display raw order book
    with st.expander("📋 Raw Order Book Data"):
        st.write("### Buy Side (BID)")
        if ob_state['buy_depth']:
            buy_df = pd.DataFrame(ob_state['buy_depth'], columns=["Price", "Quantity"])
            st.dataframe(buy_df, use_container_width=True)

        st.write("### Sell Side (ASK)")
        if ob_state['sell_depth']:
            sell_df = pd.DataFrame(ob_state['sell_depth'], columns=["Price", "Quantity"])
            st.dataframe(sell_df, use_container_width=True)

# Tab 2: Market Impact
with tab2:
    st.header("Market Impact Analysis")

    # Compute microstructure metrics
    if len(st.session_state.exchange.order_book.trades) > 0:
        trades_df = pd.DataFrame(st.session_state.exchange.order_book.trades)

        # Compute Order Flow Imbalance (OFI)
        buy_volume = trades_df[trades_df["buyer"] == "RL_AGENT"]["quantity"].sum()
        sell_volume = trades_df[trades_df["seller"] == "RL_AGENT"]["quantity"].sum()
        ofi = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-6)

        # Compute VPIN
        vpin = VPIN(bucket_size=10)
        vpin_value = vpin.compute_vpin_from_trades(st.session_state.exchange.order_book.trades)

        # Compute Kyle's Lambda (simplified)
        kyle = KyleLambda(lambda_=0.01)
        kyle_impact = kyle.estimate_impact(ofi)

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Order Flow Imbalance", f"{ofi:.4f}")
        with col2:
            st.metric("VPIN", f"{vpin_value:.4f}")
        with col3:
            st.metric("Kyle's Lambda Impact", f"{kyle_impact:.4f}")

        # Plot trade history
        if len(trades_df) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trades_df.index,
                y=trades_df["price"],
                mode="lines+markers",
                name="Trade Price",
                line=dict(color="blue"),
            ))
            fig.update_layout(
                title="Trade Price History",
                xaxis_title="Trade Index",
                yaxis_title="Price",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Plot cumulative volume
            trades_df["cumulative_volume"] = trades_df["quantity"].cumsum()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trades_df.index,
                y=trades_df["cumulative_volume"],
                mode="lines",
                name="Cumulative Volume",
                line=dict(color="purple"),
            ))
            fig.update_layout(
                title="Cumulative Trade Volume",
                xaxis_title="Trade Index",
                yaxis_title="Volume",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades executed yet. Run a simulation to see market impact metrics.")

# Tab 3: Execution Analysis
with tab3:
    st.header("Execution Algorithm Analysis")

    # Run execution algorithm
    if st.button("🚀 Run Execution Algorithm"):
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=30)

        if execution_algorithm == "TWAP":
            twap = TWAP(
                total_quantity=total_quantity,
                start_time=start_time,
                end_time=end_time,
                n_slices=n_slices,
            )
            trades = twap.execute(st.session_state.exchange, side="BUY")
            st.session_state.trade_history = trades

        elif execution_algorithm == "VWAP":
            # Generate a synthetic volume profile
            volume_profile = [100 + i * 10 for i in range(n_slices)]
            vwap = VWAP(
                total_quantity=total_quantity,
                start_time=start_time,
                end_time=end_time,
                volume_profile=volume_profile,
            )
            trades = vwap.execute(st.session_state.exchange, side="BUY")
            st.session_state.trade_history = trades

        elif execution_algorithm == "Implementation Shortfall":
            is_strategy = ImplementationShortfall(
                total_quantity=total_quantity,
                start_time=start_time,
                end_time=end_time,
                decision_price=st.session_state.exchange.get_order_book_state()["mid_price"] or 100.0,
                lambda_=0.01,
                sigma=volatility,
                risk_aversion=0.5,
                n_slices=n_slices,
            )
            trades = is_strategy.execute(st.session_state.exchange, side="BUY")
            st.session_state.trade_history = trades

        elif execution_algorithm == "Optimal Execution":
            oe_strategy = OptimalExecution(
                total_quantity=total_quantity,
                start_time=start_time,
                end_time=end_time,
                decision_price=st.session_state.exchange.get_order_book_state()["mid_price"] or 100.0,
                sigma=volatility,
                lambda_=0.01,
                eta=0.005,
                risk_aversion=0.5,
                n_slices=n_slices,
            )
            trades = oe_strategy.execute(st.session_state.exchange, side="BUY")
            st.session_state.trade_history = trades

        st.success(f"✅ Executed {len(trades)} trades using {execution_algorithm}!")

    # Display trade history
    if st.session_state.trade_history:
        trades_df = pd.DataFrame(st.session_state.trade_history)
        st.write("### Trade History")
        st.dataframe(trades_df, use_container_width=True)

        # Plot execution prices
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trades_df.index,
            y=trades_df["price"],
            mode="lines+markers",
            name="Execution Price",
            line=dict(color="orange"),
        ))
        fig.update_layout(
            title=f"{execution_algorithm} Execution Prices",
            xaxis_title="Trade Index",
            yaxis_title="Price",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Compute execution metrics
        avg_price = trades_df["price"].mean()
        total_volume = trades_df["quantity"].sum()
        price_std = trades_df["price"].std()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Execution Price", f"{avg_price:.2f}")
        with col2:
            st.metric("Total Volume Executed", f"{total_volume:,}")
        with col3:
            st.metric("Price Volatility", f"{price_std:.4f}")

# Tab 4: Simulation Control
with tab4:
    st.header("Simulation Control")

    # Generate synthetic orders
    if st.button("🎲 Generate Synthetic Orders"):
        n_orders = st.slider("Number of Orders", 10, 1000, 100, key="n_orders")

        # Update synthetic generator
        st.session_state.synthetic_generator = SyntheticMarketGenerator(
            n_traders=n_traders,
            initial_price=initial_price,
        )

        # Generate orders
        orders = st.session_state.synthetic_generator.generate_order_flow(n_orders)

        # Submit orders to exchange
        for order in orders:
            side = OrderSide.BUY if order["side"] == "BUY" else OrderSide.SELL
            st.session_state.exchange.submit_order(
                trader_id=order["trader_id"],
                side=side,
                price=order["price"],
                quantity=order["quantity"],
                order_type=OrderType.LIMIT,
            )

        st.success(f"✅ Generated and submitted {n_orders} synthetic orders!")

    # Run RL Agent
    if use_rl_agent and st.button("🤖 Run RL Agent"):
        from strategies.rl_environment import TradingEnvironment
        from strategies.rl_execution_agent import RLExecutionAgent

        # Create RL environment
        env = TradingEnvironment(
            order_book=st.session_state.exchange.order_book,
            max_steps=100,
            target_quantity=100,
        )

        # Create RL agent
        agent = RLExecutionAgent(algorithm=rl_algorithm)

        # Train for a few episodes
        st.info(f"Training {rl_algorithm} agent for 5 episodes...")
        episode_rewards = agent.train(env, episodes=5, render=False)

        st.success(f"✅ RL Agent trained! Episode rewards: {episode_rewards}")

        # Save model
        agent.save(f"{rl_algorithm}_agent.pth")
        st.info(f"Model saved as {rl_algorithm}_agent.pth")

    # Display order book history
    if st.session_state.order_book_history:
        st.write("### Order Book History")
        history_df = pd.DataFrame(st.session_state.order_book_history)
        st.dataframe(history_df, use_container_width=True)

# Run the app
if __name__ == "__main__":
    st.write("""
    ## 📌 About Q-Micro Dashboard

    This dashboard provides a **real-time visualization** of:
    - **Order Book**: Bid/ask depth, spread, and mid-price.
    - **Market Impact**: VPIN, Order Flow Imbalance, Kyle's Lambda.
    - **Execution Algorithms**: TWAP, VWAP, Implementation Shortfall, Optimal Execution.
    - **Simulation Control**: Generate synthetic orders, run RL agents.

    **How to Use:**
    1. Use the sidebar to configure market and execution parameters.
    2. Generate synthetic orders to populate the order book.
    3. Run execution algorithms to see how they perform.
    4. Analyze market impact metrics and trade history.

    **Note:** For RL training, enable the RL Agent in the sidebar and click "Run RL Agent".
    """)