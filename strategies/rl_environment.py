"""
Reinforcement Learning Environment for Q-Micro.
Implements a custom Gym-like environment for execution optimization.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from core.order_book import OrderBook
from core.order import Order, OrderSide, OrderType


@dataclass
class TradingEnvironment(gym.Env):
    """
    Custom RL environment for execution optimization.
    
    State:
    - inventory: Current inventory (positive = long, negative = short).
    - spread: Current bid-ask spread (normalized).
    - volatility: Recent volatility (rolling std of returns).
    - liquidity: Total volume at best bid/ask (normalized).
    - order_imbalance: (Buy Volume - Sell Volume) / Total Volume.
    - time_remaining: Fraction of time left in the episode.
    
    Actions:
    - 0: Buy 1 unit (aggressively)
    - 1: Sell 1 unit (aggressively)
    - 2: Buy 1 unit (passively, at bid)
    - 3: Sell 1 unit (passively, at ask)
    - 4: Wait (do nothing)
    - 5: Adjust buy size (increase)
    - 6: Adjust buy size (decrease)
    - 7: Adjust sell size (increase)
    - 8: Adjust sell size (decrease)
    
    Reward:
    - PnL from trades (execution price vs. mid-price).
    - Penalty for inventory risk (variance of inventory).
    - Penalty for execution cost (slippage).
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(
        self,
        order_book: OrderBook,
        max_inventory: int = 100,
        max_steps: int = 1000,
        target_quantity: int = 50,
        initial_price: float = 100.0,
        volatility_penalty: float = 0.1,
        inventory_penalty: float = 0.01,
    ):
        super().__init__()
        
        self.order_book = order_book
        self.max_inventory = max_inventory
        self.max_steps = max_steps
        self.target_quantity = target_quantity
        self.initial_price = initial_price
        self.volatility_penalty = volatility_penalty
        self.inventory_penalty = inventory_penalty
        
        # Action space: 9 discrete actions
        self.action_space = spaces.Discrete(9)
        
        # State space: [inventory, spread, volatility, liquidity, order_imbalance, time_remaining]
        self.observation_space = spaces.Box(
            low=np.array([-max_inventory, 0, 0, 0, -1, 0]),
            high=np.array([max_inventory, 1, np.inf, 1, 1, 1]),
            dtype=np.float32,
        )
        
        # Initialize state
        self.inventory = 0
        self.current_step = 0
        self.episode_reward = 0.0
        self.price_history = [initial_price]
        self.trade_history = []
        
        # Action mapping
        self.action_map = {
            0: {"type": "market", "side": OrderSide.BUY, "size": 1},
            1: {"type": "market", "side": OrderSide.SELL, "size": 1},
            2: {"type": "limit", "side": OrderSide.BUY, "size": 1},
            3: {"type": "limit", "side": OrderSide.SELL, "size": 1},
            4: {"type": "wait", "side": None, "size": 0},
            5: {"type": "adjust", "side": OrderSide.BUY, "size": +1},
            6: {"type": "adjust", "side": OrderSide.BUY, "size": -1},
            7: {"type": "adjust", "side": OrderSide.SELL, "size": +1},
            8: {"type": "adjust", "side": OrderSide.SELL, "size": -1},
        }
        
        # Execution sizes (can be adjusted)
        self.buy_size = 1
        self.sell_size = 1
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.inventory = 0
        self.current_step = 0
        self.episode_reward = 0.0
        self.price_history = [self.initial_price]
        self.trade_history = []
        self.buy_size = 1
        self.sell_size = 1
        
        # Reset order book (optional: can be replaced with a new one)
        self.order_book.buy_orders = {}
        self.order_book.sell_orders = {}
        self.order_book.trades = []
        
        # Add initial liquidity
        self._add_initial_liquidity()
        
        return self._get_observation(), {}
    
    def _add_initial_liquidity(self) -> None:
        """Add initial liquidity to the order book."""
        # Add some initial orders around the initial price
        for i in range(5):
            bid_price = self.initial_price - 0.01 * (i + 1)
            ask_price = self.initial_price + 0.01 * (i + 1)
            
            bid_order = Order(
                trader_id="MARKET",
                side=OrderSide.BUY,
                price=bid_price,
                quantity=100,
                order_type=OrderType.LIMIT,
            )
            ask_order = Order(
                trader_id="MARKET",
                side=OrderSide.SELL,
                price=ask_price,
                quantity=100,
                order_type=OrderType.LIMIT,
            )
            
            self.order_book.add_order(bid_order)
            self.order_book.add_order(ask_order)
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation (state)."""
        best_bid = self.order_book.get_best_bid() or self.initial_price
        best_ask = self.order_book.get_best_ask() or self.initial_price
        mid_price = (best_bid + best_ask) / 2
        spread = (best_ask - best_bid) / mid_price if mid_price > 0 else 0
        
        # Compute volatility (rolling std of last 10 prices)
        if len(self.price_history) >= 2:
            returns = np.diff(self.price_history[-10:])
            volatility = np.std(returns) / mid_price if mid_price > 0 else 0
        else:
            volatility = 0
        
        # Compute liquidity (total volume at best bid/ask)
        buy_depth = self.order_book.get_depth(OrderSide.BUY, levels=1)
        sell_depth = self.order_book.get_depth(OrderSide.SELL, levels=1)
        total_liquidity = (buy_depth[0][1] if buy_depth else 0) + (sell_depth[0][1] if sell_depth else 0)
        liquidity = min(total_liquidity / 1000, 1.0)  # Normalize
        
        # Compute order flow imbalance
        buy_volume = buy_depth[0][1] if buy_depth else 0
        sell_volume = sell_depth[0][1] if sell_depth else 0
        total_volume = buy_volume + sell_volume
        order_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
        
        # Time remaining
        time_remaining = 1.0 - (self.current_step / self.max_steps)
        
        return np.array([
            self.inventory / self.max_inventory,  # Normalized inventory
            spread,
            volatility,
            liquidity,
            order_imbalance,
            time_remaining,
        ], dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        action_info = self.action_map[action]
        reward = 0.0
        done = False
        truncated = False
        
        # Execute action
        if action_info["type"] == "market":
            reward = self._execute_market_order(action_info["side"])
        elif action_info["type"] == "limit":
            reward = self._execute_limit_order(action_info["side"])
        elif action_info["type"] == "adjust":
            self._adjust_size(action_info["side"], action_info["size"])
        # else: wait (do nothing)
        
        # Update step count
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        
        # Compute additional penalties
        inventory_penalty = self.inventory_penalty * (self.inventory ** 2)
        reward -= inventory_penalty
        
        # Update episode reward
        self.episode_reward += reward
        
        # Get observation
        observation = self._get_observation()
        
        # Info dict
        info = {
            "inventory": self.inventory,
            "spread": observation[1],
            "volatility": observation[2],
            "liquidity": observation[3],
            "order_imbalance": observation[4],
            "time_remaining": observation[5],
            "episode_reward": self.episode_reward,
        }
        
        return observation, reward, done, truncated, info
    
    def _execute_market_order(self, side: OrderSide) -> float:
        """Execute a market order and compute reward."""
        size = self.buy_size if side == OrderSide.BUY else self.sell_size
        
        # Get current best price
        best_bid = self.order_book.get_best_bid() or self.initial_price
        best_ask = self.order_book.get_best_ask() or self.initial_price
        mid_price = (best_bid + best_ask) / 2
        
        # Execute market order
        if side == OrderSide.BUY:
            execution_price = best_ask
            self.inventory += size
        else:
            execution_price = best_bid
            self.inventory -= size
        
        # Compute PnL (vs. mid-price)
        pnl = (mid_price - execution_price) * size if side == OrderSide.BUY else (execution_price - mid_price) * size
        
        # Update price history
        self.price_history.append(execution_price)
        
        # Record trade
        self.trade_history.append({
            "side": "BUY" if side == OrderSide.BUY else "SELL",
            "quantity": size,
            "price": execution_price,
            "inventory": self.inventory,
        })
        
        return pnl
    
    def _execute_limit_order(self, side: OrderSide) -> float:
        """Execute a limit order and compute reward."""
        size = self.buy_size if side == OrderSide.BUY else self.sell_size
        
        # Get current best price
        best_bid = self.order_book.get_best_bid() or self.initial_price
        best_ask = self.order_book.get_best_ask() or self.initial_price
        mid_price = (best_bid + best_ask) / 2
        
        # Place limit order at best bid/ask
        if side == OrderSide.BUY:
            price = best_bid
        else:
            price = best_ask
        
        # Create and add order to the book
        order = Order(
            trader_id="RL_AGENT",
            side=side,
            price=price,
            quantity=size,
            order_type=OrderType.LIMIT,
        )
        self.order_book.add_order(order)
        
        # For simplicity, assume the order is filled immediately (in practice, use MatchingEngine)
        # Here, we simulate a fill at the limit price
        if side == OrderSide.BUY:
            execution_price = price
            self.inventory += size
        else:
            execution_price = price
            self.inventory -= size
        
        # Compute PnL (vs. mid-price)
        pnl = (mid_price - execution_price) * size if side == OrderSide.BUY else (execution_price - mid_price) * size
        
        # Update price history
        self.price_history.append(execution_price)
        
        # Record trade
        self.trade_history.append({
            "side": "BUY" if side == OrderSide.BUY else "SELL",
            "quantity": size,
            "price": execution_price,
            "inventory": self.inventory,
        })
        
        return pnl
    
    def _adjust_size(self, side: OrderSide, delta: int) -> None:
        """Adjust the execution size."""
        if side == OrderSide.BUY:
            self.buy_size = max(1, self.buy_size + delta)
        else:
            self.sell_size = max(1, self.sell_size + delta)
    
    def render(self, mode: str = "human") -> None:
        """Render the environment (optional)."""
        if mode == "human":
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Inventory: {self.inventory}")
            print(f"Episode Reward: {self.episode_reward:.2f}")
            print(f"Order Book: {self.order_book}")
    
    def close(self) -> None:
        """Close the environment."""
        pass
