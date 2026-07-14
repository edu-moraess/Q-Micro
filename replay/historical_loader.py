"""
Carrega dados históricos salvos pelo recorder.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, List

class HistoricalLoader:
    def __init__(self, data_dir: str = "data/recorded"):
        self.data_dir = Path(data_dir)
    
    def list_sessions(self) -> List[str]:
        """Retorna IDs de sessões disponíveis."""
        if not self.data_dir.exists():
            return []
        return [d.name for d in self.data_dir.iterdir() if d.is_dir()]
    
    def load_session(self, session_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega trades e snapshots do orderbook de uma sessão."""
        session_path = self.data_dir / session_id
        trades = pd.read_parquet(session_path / "trades.parquet")
        book = pd.read_parquet(session_path / "orderbook.parquet")
        return trades, book