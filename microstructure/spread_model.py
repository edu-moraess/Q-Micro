"""
Q-Micro :: microstructure.spread_model

Spread = f(volatility, liquidity, order flow)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpreadModelParams:
    """Parâmetros do modelo reduzido de spread."""

    alpha: float = 0.01
    beta_vol: float = 2.5
    beta_liq: float = -0.8
    beta_flow: float = 1.2


def estimate_spread(
    volatility: float,
    depth: float,
    order_flow_imbalance: float,
    params: SpreadModelParams | None = None,
) -> float:
    """
    Estima o bid-ask spread.

    Parameters
    ----------
    volatility : float
        Volatilidade instantânea.

    depth : float
        Liquidez disponível próxima ao best bid/ask.

    order_flow_imbalance : float
        Desequilíbrio do fluxo de ordens no intervalo [-1, 1].

    params : SpreadModelParams
        Parâmetros do modelo.

    Returns
    -------
    float
        Spread estimado.
    """

    if params is None:
        params = SpreadModelParams()

    depth = max(float(depth), 1.0)

    spread = (
        params.alpha
        + params.beta_vol * float(volatility)
        + params.beta_liq * (1.0 / depth)
        + params.beta_flow * abs(float(order_flow_imbalance))
    )

    return max(spread, 0.0001)


@dataclass
class SpreadModel:
    """
    Wrapper orientado a objetos para manter compatibilidade
    com o restante do projeto.
    """

    params: SpreadModelParams = field(default_factory=SpreadModelParams)

    def estimate(
        self,
        volatility: float,
        depth: float,
        order_flow_imbalance: float,
    ) -> float:
        return estimate_spread(
            volatility=volatility,
            depth=depth,
            order_flow_imbalance=order_flow_imbalance,
            params=self.params,
        )

    def __call__(
        self,
        volatility: float,
        depth: float,
        order_flow_imbalance: float,
    ) -> float:
        return self.estimate(
            volatility,
            depth,
            order_flow_imbalance,
        )