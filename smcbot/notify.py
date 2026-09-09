"""Section 72 -- notifications.

Telegram if a token is configured, stdout otherwise.  Never raises: a broken
notification channel must not take the trading loop down with it.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import List, Optional

from .core.types import Setup, Trade


class Notifier:
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None,
                 echo: bool = True):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.echo = echo
        self.sent: List[str] = []
        self.failures = 0

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str) -> bool:
        self.sent.append(text)
        if self.echo:
            print(text, flush=True)
        if not self.enabled:
            return False
        try:
            data = urllib.parse.urlencode(
                {"chat_id": self.chat_id, "text": text,
                 "parse_mode": "HTML", "disable_web_page_preview": "true"}
            ).encode()
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{self.token}/sendMessage", data=data)
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode()).get("ok", False)
        except Exception:
            self.failures += 1
            return False

    # ------------------------------------------------------------------
    def trade_opened(self, setup: Setup, qty: float, risk_pct: float,
                     leverage: float, symbol: str) -> None:
        tps = " / ".join(f"{t.price:g}" for t in setup.targets) or "-"
        ml = f"{setup.ml_probability:.2f}" if setup.ml_probability is not None else "n/a"
        self.send(
            f"<b>{setup.side.value} {symbol}</b>  [{setup.meta.get('classification','')}]\n"
            f"Entry {setup.entry:g} ({setup.entry_type})\n"
            f"SL    {setup.stop:g}\n"
            f"TP    {tps}\n"
            f"RR    {setup.rr:.2f}   Lev {leverage:g}x   Risk {risk_pct:.2%}\n"
            f"Qty   {qty:g}\n"
            f"SMC   {setup.smc_score:.1f}   ML {ml}\n"
            f"Setup {setup.meta.get('kind','')} | {setup.meta.get('sweep_level','')}"
        )

    def trade_closed(self, trade: Trade) -> None:
        self.send(
            f"<b>{trade.result} {trade.symbol}</b> {trade.side.value}\n"
            f"Exit  {trade.exit_price:g} ({trade.exit_reason})\n"
            f"PnL   {trade.pnl:+.2f} USDT  ({trade.r_multiple:+.2f}R)\n"
            f"Fees  {trade.fees:.2f}  Funding {trade.funding:+.3f}\n"
            f"Equity {trade.equity_after:.2f}"
        )

    def alert(self, kind: str, message: str) -> None:
        self.send(f"[ALERT:{kind}] {message}")
