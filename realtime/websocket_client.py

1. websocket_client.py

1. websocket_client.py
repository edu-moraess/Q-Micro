import asyncio
import websockets
import json
import logging

logger = logging.getLogger(__name__)

class BinanceWebSocketClient:
    BASE_URL = "wss://stream.binance.com:9443/ws"

    def __init__(self, streams: list[str], on_message: callable):
        self.url = f"{self.BASE_URL}/{'/'.join(streams)}"
        self.on_message = on_message
        self.ws = None
        self.running = False

    async def connect(self):
        self.running = True
        while self.running:
            try:
                async with websockets.connect(self.url) as ws:
                    self.ws = ws
                    logger.info("WebSocket conectado")
                    async for msg in ws:
                        await self.on_message(json.loads(msg))
            except Exception as e:
                logger.error(f"WebSocket error: {e}, reconectando em 2s...")
                await asyncio.sleep(2)

    async def close(self):
        self.running = False
        if self.ws:
            await self.ws.close()