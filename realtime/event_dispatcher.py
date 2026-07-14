class EventDispatcher:
    def __init__(self):
        self._listeners = {}

    def subscribe(self, event_type: str, callback):
        self._listeners.setdefault(event_type, []).append(callback)

    def dispatch(self, event_type: str, data):
        for cb in self._listeners.get(event_type, []):
            cb(data)