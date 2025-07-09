# routers/sim_router.py
from src.trade.sim.simulator import TradeSimulator
from src.trade.sim.order_book import LimitOrder

class SimRouter:
    def __init__(self):
        self.engine = TradeSimulator(min_hits=8, max_hits=10)

    # live- or sim-agnostic interface ------------
    def send_limit(self, order: LimitOrder):
        self.engine.place_limit(order)

    def on_tick(self, **tick):
        self.engine.on_tick(**tick)

    def get_closed_df(self):
        import pandas as pd
        return pd.DataFrame(self.engine.closed)
    
    def get_trade_counts(self) -> dict[str, int]:
        """
        { 'wins': <int>, 'losses': <int>, 'total': <int> }
        """
        return {
            "wins"  : self.engine.wins,
            "losses": self.engine.losses,
            "total" : self.engine.wins + self.engine.losses
        }
