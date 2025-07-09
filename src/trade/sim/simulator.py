from __future__ import annotations
from datetime import datetime
from typing import Optional
from src.trade.sim.order_book import LimitOrder
from src.trade.sim.trade import Trade

class TradeSimulator:
    def __init__(self, min_hits: int, max_hits: int):
        self.order: Optional[LimitOrder] = None
        self.trade: Optional[Trade] = None
        self.closed = []
        self._id = 0
        self.min_hits, self.max_hits = min_hits, max_hits
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.last_exit_time: datetime | None = None

    # 👇 NEW — the router calls this when a limit order is ready
    def place_limit(self, order: LimitOrder):
        self.order = order

    # ────────────────────────────────────────────────────────────
    def on_tick(self, *,                           # ← 2 params removed
                price: float,
                bid: float,
                ask: float,
                sec_sm: int,
                dt,
                osc_sig):

        # ---------- stage-1 : pending limit order ----------
        if self.order:
            fill_price = self.order.check_fill(bid, ask)
            if fill_price is not None:
                self._open_trade(fill_price, dt, sec_sm)
            elif self.order.expired(sec_sm):    # ✓ honour TTL
                self.order = None               # cancel / flatten


        # ---------- stage-2 open trade ----------
        if self.trade:
            last_px = price if price is not None else (bid + ask) / 2
            self.trade.record_tick(dt=dt, bid=bid, ask=ask,
                                   last=last_px, osc_sig=osc_sig)

            exit_hit = self.trade.should_close(last_px, bid, ask)
            if exit_hit:
                reason, px = exit_hit
                self._close_trade(px, dt, reason)

    # ==================================================================
    # internals ---------------------------------------------------------
    # ==================================================================
    def _open_trade(self, entry_price: float, dt: datetime, sec_sm: int):
        """Convert the pending order ➜ active position"""
        if self.order is None:
            return
        self._id += 1
        self.trade = Trade(
            trade_id=self._id,
            side=self.order.side,
            entry_price=entry_price,
            entry_time=dt,
            sec_sm=sec_sm,  # <-- pass sec_sm to Trade
            stop_price=self.order.stop_price,
            take_profit=self.order.take_profit,
            osc_meta=self.order.osc_sig
        )
        self.trade.flag_time = getattr(self.order, "flag_time", None)  # <-- add this
        # first tick row marks trade_open_flag = 1
        self.trade._append_tick(dt=dt, bid=None, ask=None, last=entry_price,
                                osc_sig=self.order.osc_sig.get("pred_value",0),
                                open_flag=1, close_flag=0)
        self.order = None      # order consumed

    def _close_trade(self, exit_price: float, dt: datetime, reason: str):
        if self.trade is None:
            return

        # basic P/L
        pos = 1 if self.trade.side == 1 else -1
        pnl   = (exit_price - self.trade.entry) if pos==1 else (self.trade.entry - exit_price)

        # final tick marks trade_close_flag = 1
        self.trade._append_tick(
            dt=dt, bid=None, ask=None, last=exit_price,
            osc_sig=0, open_flag=0, close_flag=1
        )

        # win / loss counters
        if pnl > 0:
            self.wins += 1
        elif pnl < 0:
            self.losses += 1
        self.total_pnl += pnl
        self.last_exit_time = dt

        # save per-tick history once (optional)
        trade_df = self.trade.to_dataframe()       # keep or write to disk as needed

        # summary record for later stats
        self.closed.append({
            "trade_id"   : self.trade.trade_id,
            "side"       : self.trade.side,
            "entry_price": self.trade.entry,
            "exit_price" : exit_price,
            "pnl"        : pnl,
            "reason"     : reason,
            "entry_time" : self.trade.open_time,
            "exit_time"  : dt,
            "flag_time"  : self.trade.flag_time,
            "sec_sm"     : getattr(self.trade, "sec_sm", None),  # <-- add sec_sm
            **self.trade.meta
        })

        # reset state
        self.trade = None

