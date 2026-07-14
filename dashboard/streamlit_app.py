"""
Q-Micro :: dashboard.streamlit_app
--------------------------------------
Run with: streamlit run dashboard/streamlit_app.py
"""

import streamlit as st
import plotly.graph_objects as go

from simulation.market_simulator import MarketSimulator
from data.synthetic_generator import Regime

st.set_page_config(page_title="Q-Micro Terminal", layout="wide")
st.title("Q-Micro — Market Microstructure Research Terminal")

with st.sidebar:
    st.header("Simulation Controls")
    n_steps = st.slider("Steps", 50, 2000, 300, step=50)
    start_price = st.number_input("Start price", value=100.0)
    regime = st.selectbox("Regime", [r.value for r in Regime])
    seed = st.number_input("Seed", value=42, step=1)
    run_btn = st.button("Run simulation")

if run_btn:
    sim = MarketSimulator(start_price=start_price, seed=seed)
    sim.configure_regime(Regime(regime))
    history = sim.run(n_steps)

    mids = [h["mid"] for h in history if h["mid"] is not None]
    spreads = [h["spread"] for h in history if h["spread"] is not None]
    ofis = [h["ofi"] for h in history if h["ofi"] is not None]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mid Price")
        fig = go.Figure(go.Scatter(y=mids, mode="lines"))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Spread")
        fig2 = go.Figure(go.Scatter(y=spreads, mode="lines"))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Order Flow Imbalance")
    fig3 = go.Figure(go.Scatter(y=ofis, mode="lines"))
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Order Book Snapshot (last step)")
    last = history[-1]
    st.json(last["depth"])

    st.subheader("Trade Tape (last 20)")
    tape = sim.trade_tape()
    st.dataframe(tape[-20:])
else:
    st.info("Configure parameters on the left and click **Run simulation**.")