from dataclasses import dataclass
from typing import Literal, Optional
from datetime import datetime



@dataclass
class LimitOrder:
    side: int
    limit_price: float
    stop_price: float
    take_profit: float
    sec_sm_placed: int          # when it was submitted
    ttl_seconds: int            # ← 🆕 max age
    osc_sig: dict
    flag_time: datetime
     

    def expired(self, now_sec_sm: int) -> bool:
        """True ⇢ order has aged out."""
        return (now_sec_sm - self.sec_sm_placed) > self.ttl_seconds


    def check_fill(self, bid: float, ask: float) -> Optional[float]:
        """
        Return the execution price if the limit crosses the book, else None.
        LONG fills on ask <= limit;  SHORT fills on bid >= limit.
        """
        # remove comments for a limit order, leave them for a market order
        if (self.side == 1#):
            and ask <= self.limit_price):
            return ask
        if (self.side == -1#):
            and bid >= self.limit_price):
            return bid
        return None
