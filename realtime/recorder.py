"""
Grava eventos de mercado em arquivos Parquet para replay futuro.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

class EventRecorder:
    def __init__(self, base_dir: str = "data/recorded"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.trade_buffer = []
        self.book_buffer = []
    
    def record_trade(self, trade: dict):
        self.trade_buffer.append(trade)
    
    def record_orderbook(self, ob: dict):
        self.book_buffer.append(ob)
    
    def flush(self, session_id: Optional[str] = None):
        """Salva os buffers em arquivos Parquet."""
        if not session_id:
            session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        session_dir = self.base_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        if self.trade_buffer:
            df_trades = pd.DataFrame(self.trade_buffer)
            df_trades.to_parquet(session_dir / "trades.parquet", index=False)
            self.trade_buffer.clear()
        
        if self.book_buffer:
            df_book = pd.DataFrame(self.book_buffer)
            df_book.to_parquet(session_dir / "orderbook.parquet", index=False)
            self.book_buffer.clear()
        
        return session_id