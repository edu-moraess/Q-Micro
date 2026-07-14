"""
Q-Micro :: dashboard.streamlit_app
--------------------------------------
Terminal de pesquisa em microestrutura de mercado.
Rodar com: streamlit run dashboard/streamlit_app.py
"""

import sys
import os

# Garante que a raiz do repo (Q-Micro/) esteja no sys.path — o Streamlit
# Cloud só adiciona automaticamente a pasta onde está este arquivo (dashboard/),
# então sem isso os imports de core/, data/, simulation/ etc. falham.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go

from simulation.market_simulator import MarketSimulator
from data.synthetic_generator import Regime

st.set_page_config(page_title="Q-Micro Terminal", layout="wide")
st.title("Q-Micro — Terminal de Pesquisa em Microestrutura de Mercado")
st.caption(
    "Simulador de exchange eletrônica: formação de preço, liquidez, "
    "impacto de ordens e algoritmos de execução institucional."
)

# ---------------------------------------------------------------------- #
# Sidebar — controles de simulação
# ---------------------------------------------------------------------- #
with st.sidebar:
    st.header("Parâmetros da Simulação")

    n_steps = st.slider("Número de passos (ticks)", 50, 2000, 300, step=50)
    start_price = st.number_input("Preço inicial", value=100.0, min_value=0.01)

    regime_labels = {
        "CALM": "Calmo (baixa vol.)",
        "VOLATILE": "Volátil",
        "TRENDING": "Tendência",
    }
    regime_key = st.selectbox(
        "Regime de mercado",
        options=list(regime_labels.keys()),
        format_func=lambda k: regime_labels[k],
    )

    seed = st.number_input("Seed (reprodutibilidade)", value=42, step=1)
    run_btn = st.button("▶ Rodar simulação", type="primary")

# ---------------------------------------------------------------------- #
# Execução
# ---------------------------------------------------------------------- #
if run_btn:
    with st.spinner("Simulando fluxo de ordens..."):
        sim = MarketSimulator(start_price=start_price, seed=int(seed))
        sim.configure_regime(Regime(regime_key))
        history = sim.run(n_steps)

    mids = [h["mid"] for h in history if h["mid"] is not None]
    spreads = [h["spread"] for h in history if h["spread"] is not None]
    spreads_bps = [h["spread_bps"] for h in history if h["spread_bps"] is not None]
    ofis = [h["ofi"] for h in history if h["ofi"] is not None]

    # -- KPIs de topo -------------------------------------------------- #
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Preço médio (mid)", f"{(sum(mids)/len(mids)):.2f}" if mids else "—")
    col_b.metric("Spread médio", f"{(sum(spreads)/len(spreads)):.4f}" if spreads else "—")
    col_c.metric(
        "Spread médio (bps)",
        f"{(sum(spreads_bps)/len(spreads_bps)):.1f}" if spreads_bps else "—",
    )
    col_d.metric("Nº de trades", len(sim.trade_tape()))

    st.divider()

    # -- Gráficos principais -------------------------------------------- #
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Preço Médio (Mid Price)")
        fig = go.Figure(go.Scatter(y=mids, mode="lines", line=dict(color="#2E86AB")))
        fig.update_layout(xaxis_title="Passo", yaxis_title="Preço", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Spread (Bid-Ask)")
        fig2 = go.Figure(go.Scatter(y=spreads, mode="lines", line=dict(color="#E76F51")))
        fig2.update_layout(xaxis_title="Passo", yaxis_title="Spread", height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Order Flow Imbalance (OFI)")
    fig3 = go.Figure(go.Scatter(y=ofis, mode="lines", line=dict(color="#6A4C93")))
    fig3.add_hline(y=0, line_dash="dot", line_color="gray")
    fig3.update_layout(
        xaxis_title="Passo", yaxis_title="OFI (−1 a +1)", height=300
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # -- Book e trade tape ------------------------------------------------ #
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Book de Ofertas (último passo)")
        last = history[-1]
        depth = last["depth"]
        st.write("**Bids**")
        st.dataframe(
            [{"preço": p, "quantidade": q} for p, q in depth["bids"]],
            use_container_width=True,
        )
        st.write("**Asks**")
        st.dataframe(
            [{"preço": p, "quantidade": q} for p, q in depth["asks"]],
            use_container_width=True,
        )

    with col4:
        st.subheader("Fita de Negócios (últimos 20)")
        tape = sim.trade_tape()
        st.dataframe(tape[-20:], use_container_width=True)

else:
    st.info(
        "Configure os parâmetros no menu lateral e clique em "
        "**▶ Rodar simulação** para gerar o fluxo de ordens sintético."
    )