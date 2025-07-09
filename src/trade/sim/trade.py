import pandas as pd
from datetime import datetime
from typing import Literal, Optional


class Trade:
    """One open position + its tick-by-tick history."""
    def __init__(self, *,
                 trade_id: int,
                 side: int,
                 entry_price: float,
                 entry_time: datetime,
                 sec_sm: int,  # <-- add sec_sm
                 stop_price: float,
                 take_profit: float,
                 osc_meta: dict):
        self.trade_id   = trade_id
        self.side       = side 
        self.entry      = entry_price
        self.open_time  = entry_time
        self.sec_sm     = sec_sm  # <-- store sec_sm
        self.stop_price = stop_price
        self.tp_price   = take_profit
        self.meta       = osc_meta
        self._rows: list[tuple] = [] 
        self.flag_time = None  # <-- add this

        self.ticks = pd.DataFrame(columns=[
            "trade_id","datetime","bid","ask","Last","osc_sig",
            "trade_open_flag","trade_close_flag"
        ])

    # ---------- helpers ----------
    def _append_tick(self, *, dt, bid, ask, last, osc_sig,
                     open_flag=0, close_flag=0):
        # Just collect tuples – O(1)
        self._rows.append((
            self.trade_id, dt, bid, ask, last, osc_sig,
            open_flag, close_flag
        ))

    # ---------- public  ----------
    def record_tick(self, *, dt, bid, ask, last, osc_sig):
        self._append_tick(dt=dt, bid=bid, ask=ask, last=last,
                          osc_sig=osc_sig, open_flag=0, close_flag=0)

    def should_close(self, last: float, bid: float, ask: float) -> Optional[tuple[str,float]]:
        """
        Return ("Stop" or "TakeProfit", exit_price) if a target is hit, else None.
        LONG  → evaluate on bid; SHORT → on ask.
        """
        if self.side == 1:
            if last <= self.stop_price or bid <= self.stop_price:
                return ("Stop", bid)
            if bid >= self.tp_price:
                return ("TakeProfit", bid)
        else:  # SHORT
            if last >= self.stop_price or ask >= self.stop_price:
                return ("Stop", ask)
            if ask <= self.tp_price:
                return ("TakeProfit", ask)
        return None
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            self._rows,
            columns=[
                "trade_id","datetime","bid","ask","Last","osc_sig",
                "trade_open_flag","trade_close_flag"
            ]
        )
