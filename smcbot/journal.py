"""Sections 71, 96-97 -- the trade journal and model registry.

SQLite keeps everything the spec asks to persist in one file: candles are left
to the CSV cache, but every signal, score, prediction, trade, fill and error is
recorded, and every trade is tagged with the model versions that produced it.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Dict, Iterable, List, Optional

from .core.types import Trade

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    setup_id TEXT, symbol TEXT, side TEXT,
    entry_time INTEGER, entry_price REAL, qty REAL,
    stop REAL, targets TEXT,
    exit_time INTEGER, exit_price REAL, exit_reason TEXT,
    result TEXT, pnl REAL, fees REAL, funding REAL,
    r_multiple REAL, mfe REAL, mae REAL, tp_hits TEXT,
    smc_score REAL, ml_probability REAL,
    market_state TEXT, session TEXT,
    model_version TEXT, equity_after REAL,
    features TEXT, meta TEXT,
    run_id TEXT
);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time INTEGER, setup_id TEXT, side TEXT, state TEXT,
    entry REAL, stop REAL, rr REAL,
    smc_score REAL, ml_probability REAL, decision TEXT, reason TEXT,
    breakdown TEXT, run_id TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at INTEGER, kind TEXT, symbol TEXT,
    config TEXT, metrics TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS models (
    version TEXT PRIMARY KEY,
    created_at INTEGER, kind TEXT,
    feature_names TEXT, metrics TEXT, path TEXT
);
CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time INTEGER, kind TEXT, message TEXT, context TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id);
CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_signals_run ON signals(run_id);
"""


class Journal:
    def __init__(self, path: str = "smcbot.db"):
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *exc) -> None:
        self.conn.commit()
        self.close()

    # ------------------------------------------------------------------
    def start_run(self, run_id: str, kind: str, symbol: str, config: dict,
                  notes: str = "") -> None:
        import time
        self.conn.execute(
            "INSERT OR REPLACE INTO runs(run_id, created_at, kind, symbol, config,"
            " metrics, notes) VALUES(?,?,?,?,?,?,?)",
            (run_id, int(time.time() * 1000), kind, symbol,
             json.dumps(config, default=str), "{}", notes))
        self.conn.commit()

    def finish_run(self, run_id: str, metrics: dict) -> None:
        self.conn.execute("UPDATE runs SET metrics=? WHERE run_id=?",
                          (json.dumps(metrics, default=str), run_id))
        self.conn.commit()

    def record_trade(self, trade: Trade, run_id: str = "") -> None:
        self.conn.execute(
            "INSERT INTO trades(setup_id,symbol,side,entry_time,entry_price,qty,"
            "stop,targets,exit_time,exit_price,exit_reason,result,pnl,fees,funding,"
            "r_multiple,mfe,mae,tp_hits,smc_score,ml_probability,market_state,"
            "session,model_version,equity_after,features,meta,run_id) "
            "VALUES(" + ",".join("?" * 28) + ")",
            (trade.setup_id, trade.symbol, trade.side.value, trade.entry_time,
             trade.entry_price, trade.qty, trade.stop, json.dumps(trade.targets),
             trade.exit_time, trade.exit_price, trade.exit_reason, trade.result,
             trade.pnl, trade.fees, trade.funding, trade.r_multiple, trade.mfe,
             trade.mae, json.dumps(trade.tp_hits), trade.smc_score,
             trade.ml_probability, trade.market_state, trade.session,
             trade.model_version, trade.equity_after,
             json.dumps(trade.features, default=str),
             json.dumps(trade.meta, default=str), run_id))

    def record_trades(self, trades: Iterable[Trade], run_id: str = "") -> int:
        n = 0
        for t in trades:
            self.record_trade(t, run_id)
            n += 1
        self.conn.commit()
        return n

    def record_signal(self, time_: int, setup_id: str, side: str, state: str,
                      entry: float, stop: float, rr: float, smc_score: float,
                      ml_probability: Optional[float], decision: str, reason: str,
                      breakdown: Optional[dict] = None, run_id: str = "") -> None:
        self.conn.execute(
            "INSERT INTO signals(time,setup_id,side,state,entry,stop,rr,smc_score,"
            "ml_probability,decision,reason,breakdown,run_id) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time_, setup_id, side, state, entry, stop, rr, smc_score,
             ml_probability, decision, reason,
             json.dumps(breakdown or {}, default=str), run_id))

    def record_model(self, version: str, kind: str, feature_names: List[str],
                     metrics: dict, path: str = "") -> None:
        import time
        self.conn.execute(
            "INSERT OR REPLACE INTO models(version,created_at,kind,feature_names,"
            "metrics,path) VALUES(?,?,?,?,?,?)",
            (version, int(time.time() * 1000), kind, json.dumps(feature_names),
             json.dumps(metrics, default=str), path))
        self.conn.commit()

    def record_error(self, kind: str, message: str, context: dict = None) -> None:
        import time
        self.conn.execute(
            "INSERT INTO errors(time,kind,message,context) VALUES(?,?,?,?)",
            (int(time.time() * 1000), kind, message,
             json.dumps(context or {}, default=str)))
        self.conn.commit()

    # ------------------------------------------------------------------
    def stats(self, run_id: Optional[str] = None) -> Dict[str, float]:
        where, args = ("WHERE run_id=?", (run_id,)) if run_id else ("", ())
        row = self.conn.execute(
            f"SELECT COUNT(*) n, SUM(pnl) pnl, "
            f"SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) wins, "
            f"AVG(r_multiple) avg_r FROM trades {where}", args).fetchone()
        n = row["n"] or 0
        return {"trades": n, "net_pnl": round(row["pnl"] or 0.0, 4),
                "win_rate": round((row["wins"] or 0) / n, 4) if n else 0.0,
                "avg_r": round(row["avg_r"] or 0.0, 4)}

    def recent_trades(self, limit: int = 20, run_id: Optional[str] = None
                      ) -> List[sqlite3.Row]:
        where, args = ("WHERE run_id=?", (run_id,)) if run_id else ("", ())
        return list(self.conn.execute(
            f"SELECT * FROM trades {where} ORDER BY entry_time DESC LIMIT ?",
            (*args, limit)))
