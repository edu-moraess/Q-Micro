"""
Interface de controle do replay para o dashboard.
"""
from .replay_engine import ReplayEngine
from .historical_loader import HistoricalLoader

class PlaybackController:
    def __init__(self, data_dir: str = "data/recorded"):
        self.loader = HistoricalLoader(data_dir)
        self.engine: Optional[ReplayEngine] = None
    
    def load_session(self, session_id: str):
        trades, book = self.loader.load_session(session_id)
        self.engine = ReplayEngine(trades, book)
        return self.engine
    
    def list_sessions(self):
        return self.loader.list_sessions()