project_root/
│
├── trade_order.py          ← NEW decision / routing hub
│
├── routers/
│   ├── __init__.py
│   ├── sim_router.py       ← sends orders to the back-test engine (default)
│   └── tradovate_router.py ← placeholder for live trading
│
└── sim/
    ├── __init__.py
    ├── order_book.py
    ├── trade.py
    └── simulator.py
