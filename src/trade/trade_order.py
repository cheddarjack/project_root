# trade_order.py  (hub / decision layer)
from datetime import datetime
from typing import Literal
from src.trade.sim.order_book import LimitOrder
from src.trade.routers.sim_router import SimRouter         # swap to TradovateRouter later

router        = SimRouter()                     # one per back-test


def on_tick(*,
            config: dict,
            price: float,
            bid: float,
            ask: float, 
            sec_sm: int,
            dt: datetime,
            uid: int,
            osc_sig: dict,
            order_ttl_seconds: int):
    """
    Called once per tick by backtest_runner.py
    ───────────────────────────────────────────
    Only extracts what it needs from osc_sig, then
    1) maybe queues a new LimitOrder
    2) hands the tick to the sim engine so open orders / trades move forward
    """
    prev_exit_time = router.engine.last_exit_time   # fast scalar lookup


    prev_exit_time = None
    trade_blocked = in_blackout(sec_sm)
    # --- Block new trade if less than 10 minutes since last exit ---
    if prev_exit_time is not None:
        # Ensure both are datetime objects
        if isinstance(prev_exit_time, str):
            from dateutil import parser
            prev_exit_time = parser.parse(prev_exit_time)
        if (dt - prev_exit_time).total_seconds() < config['Order']['min_time_since_exit']:
            side = 0

    # ---- 1. maybe send a NEW limit order ---------------------------------
    side       : int   = osc_sig.get("side", 0)
    hit_num   : int    = config['Order']['hit_num']  # how many hits to trigger a trade
    best_val  : int  = osc_sig.get("best_val", 0)
    sec_sm_diff : int    = osc_sig.get("sec_sm_diff", 0)
    amp        : float  = osc_sig.get("amp", 0.0)
    limit      : float  = osc_sig.get("limit", price)
    slice_avg  : float  = osc_sig.get("slice_avg", 0.0)
    med_osc   : float  = osc_sig.get("med_osc", 0.0)
    best_idx  : int    = osc_sig.get("best_idx", 0)
    min_idx_seconds = config['Order']['min_idx_seconds']  # seconds
    min_idx = config['Order']['min_idx']
    med_osc_flux = config['Order']['med_osc_fluctuation']
    sma_flux = config['Order']['sma_fluctuation']


    if (not trade_blocked 
            and (side == 1 or side ==  -1) 
            and hit_num == best_val
            and sec_sm_diff >= min_idx_seconds
            and best_idx > min_idx): # and amp > 0.75:
        # print(f"sma: {sma} price: {price} limit: {limit}")
        order = LimitOrder(
            side         = side,
            limit_price  = osc_sig["limit"],          # entry
            stop_price   = osc_sig["stop_loss"],      # exit-stop
            take_profit  = osc_sig["take_profit"],    # exit-tp
            sec_sm_placed= sec_sm,
            ttl_seconds  = order_ttl_seconds,         # max age
            osc_sig      = osc_sig,                   # keep meta for debug
            flag_time    = dt                         # <-- pass flag_time
        )
        router.send_limit(order)

    # ---- 2. advance the sim engine one tick -------------------------------
    router.on_tick(
        price      = price,
        bid        = bid,
        ask        = ask,
        sec_sm     = sec_sm,
        dt         = dt,
        osc_sig    = osc_sig
    )

# helper for your report at the end of a run
def get_closed_trades_df():
    return router.get_closed_df()

# trade_order.py
def get_trade_counts():
    return router.get_trade_counts()

# trade_order.py  – insert near the top, right after your imports
# --------------------------------------------------------------

# ✦ No-trade windows expressed in "seconds since midnight"
#    [12:15–12:45]  [13:15–13:45]  [20:25–21:10]
BLACKOUTS = [
    (12*3600 + 15*60, 12*3600 + 45*60),   # 44100 – 45900
    (13*3600 + 15*60, 13*3600 + 45*60),   # 47700 – 49500
    (20*3600 + 25*60, 21*3600 + 10*60),   # 73500 – 76200
]

def in_blackout(sec_sm: int) -> bool:
    return any(start <= sec_sm < end for start, end in BLACKOUTS)

