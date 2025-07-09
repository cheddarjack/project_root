"""
Replace every PASS with real Tradovate REST-/WebSocket calls when you go live.
"""
from . import BaseRouter
from sim.order_book import LimitOrder

class TradovateRouter(BaseRouter):
    def __init__(self, api_key: str):
        self.api_key = api_key
        # TODO: open WebSocket, auth, etc.

    def send_limit(self, order: LimitOrder):
        # TODO: translate LimitOrder → Tradovate JSON and POST it
        pass
