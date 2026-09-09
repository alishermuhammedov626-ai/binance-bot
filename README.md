# SMC Futures Bot

Smart-Money-Concept futures bot for a single crypto pair, built to the 100-point
specification: liquidity → sweep → structure shift → M5 setup → M1 entry →
structure-based stop → liquidity-based target → ML quality filter → risk gate.

The headline requirement was that **the backtester must not see the future**.
That is not a claim here, it is a tested property — see
[Proof it cannot see the future](#proof-it-cannot-see-the-future).

```
LIQUIDITY → SWEEP → STRUCTURE SHIFT → M5 CONFIRMATION → M1 CONFIRMATION
         → ZONE RETEST → RR → SMC SCORE → ML FILTER → RISK → EXECUTION
```

Pure Python 3.9+, **no dependencies**. LightGBM / scikit-learn are used
automatically if installed, but nothing requires them.

---

## Quick start

```bash
# 1. Everything works offline on generated data first
python -m smcbot backtest --synthetic --days 60 --rejections

# 2. Real data (Binance USD-M futures, public endpoint)
python -m smcbot fetch --symbol BTCUSDT --days 365 --out data/btc.csv

# 3. The full section-91 pipeline:
#    in-sample → out-of-sample → walk-forward ML → Monte Carlo → verdict
python -m smcbot pipeline --csv data/btc.csv --journal runs/smcbot.db
```

Other commands: `dataset`, `walkforward`, `train`, `montecarlo`, `dashboard`,
`config`. Any command takes `--set key.path=value` overrides, e.g.
`--set risk.min_rr=2.0 ml.threshold=0.6`.

### Single-file build

```bash
python scripts/build_dist.py                # -> dist/smcbot.pyz + tar.gz + sample CSV
python dist/smcbot.pyz backtest --csv data/btc.csv
```

`zipapp` archives the real package rather than concatenating sources, so the
bundle is exactly the code the test suite runs.

### CSV format

`open_time,open,high,low,close,volume` — M1 candles, `open_time` in
milliseconds (or ISO-8601). A header row is optional. Input is sorted,
de-duplicated and checked for impossible OHLC before use, and gaps are
reported rather than silently interpolated.

```
open_time,open,high,low,close,volume
1704067200000,42000.0,42007.1,41993.2,42000.2,50.37
```

---

## Proof it cannot see the future

Look-ahead bias is what makes a backtest lie. Three mechanisms prevent it, and
`tests/test_no_lookahead.py` verifies all three.

**1. The data layer makes leakage structurally impossible.**
Candles enter as a stream of closed M1 bars. M5 and M15 are *aggregated* from
that stream and a bar is published only once its final M1 constituent closes —
a partially formed M15 bar is never visible. Daily, weekly and session extremes
roll over only when the period actually ends.

**2. Every object records when it became *knowable*, not when it happened.**
A swing high at bar `i` needs `right` bars to its right, which do not exist at
bar `i`. It is therefore published at bar `i + right`, and `Swing.confirmed_at`
records that. Downstream code gates on `confirmed_at`, never on `Swing.time`.
Liquidity levels, sweeps and order blocks carry the same distinction. Update
order within a bar is deliberate: sweeps are detected against the liquidity map
*as it stood before the bar*, and new levels are registered last, so a level can
never be created and swept by the same candle.

**3. The backtester is strictly causal.** Each bar is processed in this order:

| step | what happens |
|------|--------------|
| 1 | fill orders placed at the **previous** bar's close |
| 2 | manage the open position against this bar's OHLC |
| 3 | **only now** show the bar to the analysis engines |
| 4 | decide — any order placed goes to the **next** bar |

A signal formed on the close of bar *t* can never fill before bar *t+1*.

### How this is tested

- **Prefix determinism** — decisions made during the first K bars are identical
  whether the engine is later fed 2K bars or stopped at K.
- **Future mutation** — every candle after a cut is replaced with a completely
  different (crashing, high-volatility) path. Every trade *opened* at or before
  the cut must be byte-identical, as must the equity curve. If anything peeked
  ahead, a rewritten future would change the past.
- **Structural invariants** — no aggregated bar is ever partial, no swing
  confirms before its right-hand bars, no object is consulted before its
  `confirmed_at`.
- **A meta-test that the detector is not vacuous.** A deliberately cheating
  backtester peeks at the next four hours before committing to an order; the
  mutation test must catch it. It does. (Writing this test caught a real
  weakness: cuts placed after a trade *closed* let the planted leak hide,
  because the peek's verdict was already settled by pre-cut candles. Cuts are
  now placed just after each trade *opens*.)

Labels are the one legitimate use of future data (section 49): after the replay
is finished, `ml/dataset.py` walks forward to see whether TP or SL came first.
Features are built live during the replay and never touch it.

---

## Running on a server

```bash
git clone <repo> /opt/smcbot && cd /opt/smcbot
./scripts/server_backtest.sh BTCUSDT 365
```

That single command downloads the candles (resumably), runs the full pipeline
and writes everything into `runs/<symbol>-<timestamp>/`:

| file | contents |
|------|----------|
| `log.txt` | the whole console output |
| `metrics.json` | machine-readable results |
| `smcbot.db` | SQLite journal — trades, signals, models, errors |
| `config.json` | the exact config that run used |

Exit code **0** means every production check passed, **2** means they did not
(a normal result, not a crash), **1** means the run itself failed.

### What it costs

Measured on this codebase, single core:

| workload | time | peak RAM |
|----------|------|----------|
| backtest, 120 days (173k bars) | 83 s | 131 MB |
| backtest, 1 year (526k bars) | ~4 min | ~130 MB |
| full `pipeline`, 1 year (4 passes + ML) | ~20–30 min | ~200 MB |
| fetching 1 year of M1 | ~2–5 min | negligible |

Memory is bounded by ring buffers, so it does **not** grow with history
length — a 3-year run uses the same RAM as a 3-month one. **A 1 GB / 1 vCPU VPS
is enough.** The work is pure CPU on one core; more cores only help if you run
several symbols or configs in parallel.

### Detached runs

```bash
nohup ./scripts/server_backtest.sh BTCUSDT 365 > /dev/null 2>&1 &
tail -f runs/BTCUSDT-*/log.txt
```

### Scheduled runs (systemd)

```bash
sudo useradd -r -s /usr/sbin/nologin -d /opt/smcbot smcbot
sudo chown -R smcbot:smcbot /opt/smcbot
sudo cp deploy/smcbot-backtest.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now smcbot-backtest.timer
journalctl -u smcbot-backtest -f
```

The unit runs as an unprivileged user with `ProtectSystem=strict` and write
access to nothing but `data/` and `runs/`. A backtest needs no more than that.

### Docker

```bash
docker compose -f deploy/docker-compose.yml build
docker compose -f deploy/docker-compose.yml run --rm backtest \
    pipeline --csv data/BTCUSDT_1m.csv
```

The image builds on `python:3.12-slim` with no pip install (there is nothing to
install) and runs the engine tests during the build, so a broken build fails
instead of shipping a container that starts but miscalculates.

### Downloading a year of data

`fetch` paces itself and survives interruptions, which matters unattended:
a year of M1 is ~350 requests.

```bash
python -m smcbot fetch --symbol BTCUSDT --days 365 --out data/btc.csv --resume
```

It writes each batch to disk as it arrives, watches the exchange's
`X-MBX-USED-WEIGHT-1M` header and pauses before the IP limit, retries 429/418
and transport errors with exponential backoff (honouring `Retry-After`), and
fails fast on a 4xx like a mistyped symbol instead of retrying a typo. If it
dies anyway, `--resume` continues from the last candle on disk rather than
starting over.

Gaps are **reported, never interpolated** — inventing candles to fill an
exchange outage would quietly corrupt the backtest.

---

## Architecture

```
smcbot/
├── config.py            every tunable parameter, JSON-serialisable
├── core/
│   ├── types.py         Candle, Swing, LiquidityLevel, Sweep, FVG, OB, Setup, Trade
│   └── series.py        streaming M1 → M5/M15 aggregation, sessions, day/week
├── engine/              "what is the market doing"
│   ├── swings.py        adaptive fractals with confirmation lag      (§5)
│   ├── liquidity.py     weekly/daily/session/TF levels, strength, clusters (§3-4)
│   ├── structure.py     external + internal structure, CHOCH/BOS, range (§6-8,13-16)
│   ├── sweep.py         manipulation detection and 0-100 scoring     (§10-11)
│   ├── displacement.py  impulse strength — a bonus, never a gate     (§12)
│   ├── zones.py         FVG, order blocks, breakers, mitigation      (§17-20)
│   └── context.py       wires it together, owns the update order
├── strategy/            "should we trade it"
│   ├── setup.py         the §67 state machine, TTL and invalidation  (§20-24,68-70)
│   ├── scoring.py       the 100-point SMC score                      (§44-45)
│   ├── targets.py       liquidity-based TP hierarchy                 (§27-29)
│   ├── risk.py          structure stops, sizing, liquidation safety  (§25-26,32-34)
│   └── decision.py      final gate: limits, cooldowns, SMC+ML        (§33-38,46,87)
├── ml/                  "is this setup any good"
│   ├── features.py      the §48 feature vector, past-only
│   ├── dataset.py       forward-simulated labels                     (§49)
│   ├── model.py         pure-Python histogram GBT (+ LightGBM/sklearn)
│   └── walkforward.py   time-series folds, embargo, leak-free predictor (§53-54)
├── backtest/            engine.py · metrics.py · montecarlo.py       (§55-61)
├── journal.py           SQLite: trades, signals, models, errors      (§71,96-97)
├── dashboard.py         live state view                              (§98)
└── notify.py            Telegram / stdout                            (§72)
```

**SMC generates signals, ML only filters them** (§47). The model never invents a
trade; it answers "how good is this setup" and can only veto.

---

## How a trade is built

1. **Liquidity map** — previous week/day high & low, daily open, session highs
   and lows, M15/M5/M1 swings, equal highs/lows. Each gets a strength
   (weekly 10 → M1 3); levels stacked within 0.25 ATR form a cluster (+7).
2. **Sweep** — wick beyond a level, close back inside, within 3 bars. Scored
   0–100 on level importance, penetration depth, wick ratio, rejection depth,
   speed and volume.
3. **Structure shift** — M15 internal CHOCH/BOS in the sweep's direction,
   confirmed on candle *close*. External structure decides BULLISH / BEARISH /
   RANGE; inside a range, only the extremes are tradable and the middle 30% is
   blocked outright.
4. **M5 confirmation** — CHOCH/BOS, then a fresh FVG / order block / breaker.
5. **Retest** — price must come back to the zone. Chasing is refused; if price
   is more than 0.8 ATR beyond the zone the setup waits or expires.
6. **M1 entry** — sweep + CHOCH/BOS. Limit at the M1 zone if price has not
   reached it, otherwise a confirmation market entry.
7. **Stop** — behind the M1 confirmation swing plus an ATR/tick buffer, falling
   back to the M5 invalidation swing if the M1 stop is unreasonably tight.
   Never a fixed percentage, never dragged into the structure.
8. **Targets** — the next resting liquidity, in hierarchy (M1 internal → M5 →
   M15 swing → daily → weekly), scored by distance, strength and RR.
9. **Gates** — RR ≥ 1.5, SMC score ≥ 70, ML ≥ threshold, plus position,
   cooldown, daily-loss, trade-count, spread, volatility and liquidation checks.
   Any single failure is NO TRADE.

Position size comes from **risk and stop distance, never from leverage**:

```
Risk_USDT    = equity × risk_pct          (0.25%–0.5%)
PositionSize = Risk_USDT / SL_distance
```

17× leverage only determines the margin required. A stop that sits too close to
the liquidation price is refused outright.

---

## Backtest realism

Taker/maker fees, slippage, 8-hourly funding, execution delay, exchange
qty/notional rounding and a liquidation model are all applied. Where a bar
could have hit both the stop and a target, **the stop is assumed first** — the
same pessimism is used when generating ML labels, so training and live results
stay consistent.

One consequence is worth stating plainly: at typical SMC stop distances, a
round trip costs roughly **0.3–0.4R in fees alone** on a 0.05% taker fee. A
strategy that looks marginally profitable before costs is usually not
profitable after them. That is the reality the backtester reports, not a bug.

Reported metrics go well beyond win rate (§57–58): profit factor, expectancy,
average win/loss, max drawdown, Sharpe, Sortino, Calmar, MFE/MAE, streaks, fees
and funding — plus per-session and per-market-state breakdowns.
**The production gate is expectancy > 0 after costs** (§59), not win rate. A 70%
win rate with oversized losses is explicitly a failing strategy.

---

## The ML filter

Features (§48) cover structure on all three timeframes, liquidity type and
strength, sweep geometry, displacement, zone quality, RR, SL distance,
volatility, session, day of week, distance from daily/weekly reference levels
and recent results — all past-only.

Training uses **expanding-window time-series folds with an embargo**, never a
random split (§53). `WalkForwardPredictor` then scores each bar with the model
whose training data ended *before* that bar, so an ML-enabled backtest stays
out-of-sample. Before the first fold it returns nothing and the bot runs on the
SMC score alone.

The model is a histogram gradient-boosted tree ensemble with logistic loss and
Newton leaf values, written in pure Python. Tree models suit this data (tabular,
mixed-scale, interaction-heavy, few samples); a neural network is not needed
(§50).

---

## Honest limitations

- **Synthetic data proves the plumbing, not the edge.** `--synthetic` exists so
  the pipeline can be exercised offline. Its sweeps are not followed by
  systematically profitable reversions, so expectancy there is roughly zero
  minus costs. Any edge claim requires real data.
- **Sample size.** The spec asks for 500–3000 trades. At the intended 2–4
  setups per day that is roughly **1–3 years of M1 data**. Conclusions from a
  few dozen trades are noise — the walk-forward output prints per-fold AUC
  precisely so early folds can be seen to be unreliable.
- **Parameters are defaults from the specification, not optimised.** This is
  deliberate (§60): tuning 100 parameters to one coin and one period is
  overfitting, not research.
- **No live order routing.** The engine, risk gates, journal, dashboard and
  notifications are complete, and `DecisionEngine` is shared between backtest
  and any live runner so the rules cannot drift apart. Placing real orders
  needs exchange credentials and the §74–76 order checklist wired to the API.
- **News filter is off** unless a real feed is supplied — the bot never assumes
  "no news" (§66).

Per §90–91, nothing should touch real money before out-of-sample plus
walk-forward plus 2–4 weeks of paper trading, and then only at 0.1–0.25% risk.

---

## Tests

```bash
python -m unittest discover -s tests -v
```

135 tests: look-ahead (15, including the meta-test that plants a leak),
engines (18), strategy, risk and backtest mechanics (40), ML (24),
download hardening (10), integration, runner and CLI (28).

---

## Configuration

```bash
python -m smcbot config --out my.json     # write the full default config
python -m smcbot backtest --config my.json --set risk.risk_per_trade=0.0025
```

Key defaults: leverage 17×, risk 0.25–0.5% per trade, max 1 position, min RR
1.5, SMC threshold 70, ML threshold 0.55, daily loss limit 2%, max 5 trades/day,
cooldowns 12 min (25 after a loss, escalating to 60/180 on 3/5 consecutive
losses).
