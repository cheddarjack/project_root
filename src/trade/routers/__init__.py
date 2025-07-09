from abc import ABC, abstractmethod
from typing import Literal
from src.trade.sim.order_book import LimitOrder

Side = Literal["LONG", "SHORT"]

class BaseRouter(ABC):
    """Common interface so trade_order.py doesn’t care where orders go."""
    @abstractmethod
    def send_limit(self, order: LimitOrder): ...
