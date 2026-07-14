"""
Motor de replay: reproduz eventos históricos com controle de velocidade.
"""
import time
import pandas as pd
from typing import Optional, Callable
from datetime import datetime

class ReplayEngine:
    def __init__(self, trades: pd.DataFrame, orderbook_snapshots: pd.DataFrame):
        self.trades = trades.sort_values("timestamp")
        self.orderbook = orderbook_snapshots.sort_values("timestamp")
        self.current_index = 0
        self.playing = False
        self.speed = 1.0  # 1x = tempo real
        self.callback_trade: Optional[Callable] = None
        self.callback_book: Optional[Callable] = None
    
    def register_trade_callback(self, cb: Callable):
        self.callback_trade = cb
    
    def register_book_callback(self, cb: Callable):
        self.callback_book = cb
    
    def play(self):
        self.playing = True
    
    def pause(self):
        self.playing = False
    
    def stop(self):
        self.playing = False
        self.current_index = 0
    
    def set_speed(self, speed: float):
        self.speed = max(0.1, min(10.0, speed))
    
    def seek(self, index: int):
        self.current_index = max(0, min(index, len(self.trades)-1))
    
    def step_forward(self):
        """Avança um evento (trade) e dispara callbacks."""
        if self.current_index >= len(self.trades):
            return False
        trade = self.trades.iloc[self.current_index].to_dict()
        # Procura o snapshot do orderbook mais próximo no tempo
        ts = trade["timestamp"]
        book_rows = self.orderbook[self.orderbook["timestamp"] <= ts]
        if not book_rows.empty:
            book = book_rows.iloc[-1].to_dict()
            if self.callback_book:
                self.callback_book(book)
        if self.callback_trade:
            self.callback_trade(trade)
        self.current_index += 1
        return True
    
    def run_playback_loop(self):
        """Loop de reprodução contínua (chamar em thread separada)."""
        while self.playing:
            if self.current_index >= len(self.trades):
                self.playing = False
                break
            if self.speed > 0:
                self.step_forward()
                # Aguarda o tempo real entre trades (ajustado pela velocidade)
                if self.current_index < len(self.trades):
                    current_ts = self.trades.iloc[self.current_index-1]["timestamp"]
                    next_ts = self.trades.iloc[self.current_index]["timestamp"]
                    delay = (next_ts - current_ts).total_seconds() / self.speed
                    if delay > 0:
                        time.sleep(min(delay, 0.1))  # cap para evitar travamentos longos
            else:
                time.sleep(0.01)