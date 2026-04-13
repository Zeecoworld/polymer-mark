# -*- coding: utf-8 -*-
"""
Polymarket AI Paper Trading Bot
Strategy : Replicate (Llama 3.3 70B) NLU + Spread Arbitrage + Risk Management
Mode     : PAPER TRADE ONLY  -  no real money moves
Author   : github.com/zeecomedia

SETUP
-----
1. pip install httpx
2. Get free token at https://replicate.com  (sign up, no card needed for $5 credit)
3. Paste your token in REPLICATE_API_KEY below (line 22)
4. python bot.py

DASHBOARD
---------
In a second terminal:  python -m http.server 8080
Then open:             http://localhost:8080/dashboard.html
"""

import asyncio
import json
import random
import httpx
import os
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional

# ==============================================================================
#  CONFIG  -  edit these
# ==============================================================================

from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env

REPLICATE_API_KEY  = os.getenv("REPLICATE_API_KEY")          # <-- paste your token here
REPLICATE_MODEL    = "meta/meta-llama-3-70b-instruct"
REPLICATE_API_URL  = "https://api.replicate.com/v1/models/{model}/predictions"

PAPER_BALANCE_USDC  = 1000.0   # starting paper balance
GAMMA_API           = "https://gamma-api.polymarket.com"
SCAN_INTERVAL       = 30       # seconds between scans
MAX_POSITION_PCT    = 0.05     # max 5% of balance per trade
MIN_EDGE_PCT        = 0.04     # minimum edge to enter (4 cents on $1)
MAX_OPEN_POSITIONS  = 8
LOG_FILE            = "paper_trades.json"


# ==============================================================================
#  DATA MODELS
# ==============================================================================
@dataclass
class Market:
    id:          str
    question:    str
    yes_price:   float
    no_price:    float
    volume:      float
    end_date:    str
    category:    str = "misc"
    description: str = ""


@dataclass
class Position:
    market_id:         str
    question:          str
    side:              str    # YES | NO
    shares:            float
    entry_price:       float
    current_price:     float
    timestamp:         str
    claude_confidence: float
    edge:              float
    status:            str   = "OPEN"   # OPEN | CLOSED
    pnl:               float = 0.0
    close_price:       float = 0.0
    close_time:        str   = ""


@dataclass
class BotState:
    balance:          float = PAPER_BALANCE_USDC
    total_pnl:        float = 0.0
    positions:        list  = field(default_factory=list)
    closed_trades:    list  = field(default_factory=list)
    scan_count:       int   = 0
    signals_analyzed: int   = 0
    trades_taken:     int   = 0
    start_time:       str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    log:              list  = field(default_factory=list)


state = BotState()


# ==============================================================================
#  POLYMARKET GAMMA API  (2026 structure)
# ==============================================================================
async def fetch_markets(limit: int = 50) -> list[Market]:
    """
    Fetch active markets sorted by 24h volume (highest first).
    Prices come in outcomePrices as a JSON string: "[0.65, 0.35]"
    index 0 = YES, index 1 = NO
    """
    try:
        async with httpx.AsyncClient(timeout=15) as http:
            r = await http.get(
                f"{GAMMA_API}/markets",
                params={
                    "active":    "true",
                    "closed":    "false",
                    "limit":     limit,
                    "order":     "volume24hr",
                    "ascending": "false",
                }
            )
            r.raise_for_status()
            items = r.json()
            if not isinstance(items, list):
                items = items.get("markets", [])

            markets = []
            for m in items:
                try:
                    # --- current 2026 price structure ---
                    op  = m.get("outcomePrices")
                    out = m.get("outcomes", '["Yes","No"]')

                    if op:
                        prices   = json.loads(op)  if isinstance(op,  str) else op
                        outcomes = json.loads(out) if isinstance(out, str) else out
                        yi = next((i for i, o in enumerate(outcomes) if o.lower() in ("yes","true")),  0)
                        ni = next((i for i, o in enumerate(outcomes) if o.lower() in ("no", "false")), 1)
                        yes_p, no_p = float(prices[yi]), float(prices[ni])
                    else:
                        # fallback: older tokens[] structure
                        toks  = m.get("tokens", [])
                        y_tok = next((t for t in toks if str(t.get("outcome","")).upper() == "YES"), None)
                        n_tok = next((t for t in toks if str(t.get("outcome","")).upper() == "NO"),  None)
                        if not y_tok or not n_tok:
                            continue
                        yes_p, no_p = float(y_tok["price"]), float(n_tok["price"])

                    if yes_p <= 0 or no_p <= 0:
                        continue

                    markets.append(Market(
                        id          = str(m.get("id") or m.get("conditionId", "")),
                        question    = m.get("question", ""),
                        yes_price   = yes_p,
                        no_price    = no_p,
                        volume      = float(m.get("volume24hr") or m.get("volume") or 0),
                        end_date    = m.get("endDate", ""),
                        category    = m.get("category", "misc"),
                        description = (m.get("description") or "")[:400],
                    ))
                except Exception:
                    continue

            log_event(f"[API] Fetched {len(markets)} markets from Gamma API")
            return markets

    except Exception as e:
        log_event(f"[API ERROR] {e} -- using mock data")
        return _mock_markets()


def _mock_markets() -> list[Market]:
    """Fallback synthetic markets when the API is unreachable."""
    base = [
        ("Will BTC close above $90k on Apr 15?",     0.42, 0.58, 2_100_000, "crypto"),
        ("Will ETH hit $4k before May 1?",            0.31, 0.69,   890_000, "crypto"),
        ("NBA: Lakers to win next game?",             0.55, 0.45,   340_000, "sports"),
        ("Will Fed cut rates in May 2026?",           0.38, 0.62, 1_200_000, "finance"),
        ("Will NVDA stock be above $900 by May?",    0.61, 0.39,   560_000, "finance"),
        ("EPL: Arsenal to win vs Chelsea?",           0.48, 0.52,   230_000, "sports"),
        ("Will GPT-5 be announced before June?",     0.44, 0.56,   410_000, "tech"),
        ("US inflation below 3% in April report?",   0.52, 0.48,   780_000, "finance"),
        ("Will Solana flip Ethereum by market cap?", 0.12, 0.88,   150_000, "crypto"),
        ("NBA Finals: Will Celtics win 2026?",        0.28, 0.72, 1_900_000, "sports"),
    ]
    now = datetime.now(timezone.utc)
    return [
        Market(
            id=f"mock_{i}", question=q,
            yes_price=round(yp + random.uniform(-0.02, 0.02), 3),
            no_price =round(np + random.uniform(-0.02, 0.02), 3),
            volume=vol,
            end_date=(now + timedelta(days=random.randint(3, 30))).isoformat(),
            category=cat,
        )
        for i, (q, yp, np, vol, cat) in enumerate(base)
    ]


# ==============================================================================
#  REPLICATE  /  LLAMA 3.3 70B
# ==============================================================================
async def call_replicate_llama(prompt: str) -> str:
    """
    Call meta/llama-3.3-70b-instruct on Replicate.
    Uses Llama 3 chat template to embed system instructions -- no system_prompt field.
    Prefer:wait header returns output synchronously (no polling needed).
    """
    system = (
        "You are a quantitative prediction market analyst. "
        "Always respond with valid JSON only. No markdown fences, no extra text outside the JSON."
    )
    # Llama 3 chat template
    formatted = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    headers = {
        "Authorization": f"Bearer {REPLICATE_API_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "wait",
    }
    payload = {
        "input": {
            "prompt":      formatted,
            "max_tokens":  300,
            "temperature": 0.2,
            "stop":        ["<|eot_id|>", "<|end_of_text|>"],
        }
    }
    url = REPLICATE_API_URL.format(model=REPLICATE_MODEL)

    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.post(url, headers=headers, json=payload)
        r.raise_for_status()
        result = r.json()

        output = result.get("output")
        if isinstance(output, list):
            return "".join(output).strip()
        if isinstance(output, str):
            return output.strip()

        # fallback polling (cold boot)
        poll_url = result.get("urls", {}).get("get", "")
        if not poll_url:
            raise ValueError(f"No output and no poll URL. Response: {result}")

        for _ in range(30):
            await asyncio.sleep(1)
            poll = await http.get(poll_url, headers=headers)
            poll.raise_for_status()
            data = poll.json()
            if data.get("status") == "succeeded":
                out = data.get("output", [])
                return ("".join(out) if isinstance(out, list) else out).strip()
            if data.get("status") in ("failed", "canceled"):
                raise ValueError(f"Replicate failed: {data.get('error')}")

        raise TimeoutError("Replicate timed out after 30s")


async def analyse_market(market: Market) -> Optional[dict]:
    """
    Ask Llama to estimate the TRUE probability vs the market-implied price.
    Returns dict with side, entry_price, true_prob, confidence, edge, reasoning.
    """
    prompt = (
        f"Analyse this Polymarket prediction market contract.\n\n"
        f"Question : {market.question}\n"
        f"Category : {market.category}\n"
        f"YES price: {market.yes_price:.3f}  (market implies {market.yes_price*100:.1f}% probability)\n"
        f"NO  price: {market.no_price:.3f}\n"
        f"Volume 24h: ${market.volume:,.0f}\n"
        f"Description: {market.description or 'N/A'}\n\n"
        f"Estimate the TRUE probability of YES based on your knowledge.\n\n"
        f'Respond ONLY with this exact JSON (no markdown, no extra text):\n'
        f'{{"true_prob_yes": 0.55, "confidence": 0.70, "reasoning": "one sentence", "data_quality": "HIGH"}}\n\n'
        f"data_quality: HIGH = well-known event, MEDIUM = some uncertainty, LOW = obscure"
    )

    try:
        text = await call_replicate_llama(prompt)

        # strip accidental markdown fences
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]

        data      = json.loads(text)
        true_prob = max(0.01, min(0.99, float(data.get("true_prob_yes", market.yes_price))))
        conf      = max(0.01, min(0.99, float(data.get("confidence",    0.5))))
        reasoning = data.get("reasoning", "")
        dq        = data.get("data_quality", "MEDIUM")

        edge_yes = true_prob - market.yes_price
        edge_no  = (1 - true_prob) - market.no_price

        if edge_yes >= edge_no:
            return dict(side="YES", entry_price=market.yes_price,
                        true_prob=true_prob, confidence=conf,
                        edge=edge_yes, reasoning=reasoning, data_quality=dq)
        else:
            return dict(side="NO", entry_price=market.no_price,
                        true_prob=1 - true_prob, confidence=conf,
                        edge=edge_no, reasoning=reasoning, data_quality=dq)

    except Exception as e:
        log_event(f"[LLM ERROR] {market.question[:45]}: {e}")
        return None


# ==============================================================================
#  SPREAD ARBITRAGE  (free, no LLM call)
# ==============================================================================
def detect_mispricing(market: Market) -> Optional[dict]:
    """Flag markets where YES+NO prices don't sum to ~1.00 (price lag anomaly)."""
    spread = market.yes_price + market.no_price
    if spread < 0.94:
        gap  = 1.0 - spread
        side = "YES" if market.yes_price < market.no_price else "NO"
        return {"type": "spread_arb", "gap": gap, "side": side}
    return None


# ==============================================================================
#  RISK MANAGER
# ==============================================================================
def risk_check(edge: float, confidence: float, market: Market) -> tuple[bool, str]:
    open_positions = [p for p in state.positions if p.status == "OPEN"]
    if len(open_positions) >= MAX_OPEN_POSITIONS:
        return False, f"Max open positions ({MAX_OPEN_POSITIONS}) reached"
    if edge < MIN_EDGE_PCT:
        return False, f"Edge {edge:.3f} below min {MIN_EDGE_PCT}"
    if confidence < 0.45:
        return False, f"Confidence {confidence:.2f} too low"
    if market.volume < 10_000:
        return False, f"Volume ${market.volume:,.0f} below $10k minimum"
    if market.id in {p.market_id for p in open_positions}:
        return False, "Already have open position in this market"
    return True, "OK"


def position_size(edge: float, confidence: float) -> float:
    """Half-Kelly sizing capped at MAX_POSITION_PCT of balance."""
    kelly = edge * confidence
    return min(state.balance * kelly, state.balance * MAX_POSITION_PCT, state.balance * 0.10)


# ==============================================================================
#  PAPER EXECUTION
# ==============================================================================
def open_position(market: Market, signal: dict) -> Optional[Position]:
    size = position_size(signal["edge"], signal["confidence"])
    if size < 1.0 or size > state.balance:
        return None

    shares = size / signal["entry_price"]
    state.balance -= size
    pos = Position(
        market_id         = market.id,
        question          = market.question,
        side              = signal["side"],
        shares            = shares,
        entry_price       = signal["entry_price"],
        current_price     = signal["entry_price"],
        timestamp         = datetime.now(timezone.utc).isoformat(),
        claude_confidence = signal["confidence"],
        edge              = signal["edge"],
    )
    state.positions.append(pos)
    state.trades_taken += 1
    log_event(
        f"[OPEN] {signal['side']} {market.question[:50]} | "
        f"edge={signal['edge']:.3f} size=${size:.2f} conf={signal['confidence']:.2f}"
    )
    return pos


def simulate_price_drift(pos: Position, markets: list[Market]) -> float:
    live = next((m for m in markets if m.id == pos.market_id), None)
    if live:
        return live.yes_price if pos.side == "YES" else live.no_price
    drift   = random.gauss(0.002, 0.015)
    new_p   = pos.current_price + drift
    return max(0.01, min(0.99, new_p))


def update_positions(markets: list[Market]):
    for pos in state.positions:
        if pos.status != "OPEN":
            continue
        pos.current_price = simulate_price_drift(pos, markets)
        pos.pnl = (pos.current_price - pos.entry_price) * pos.shares
        if pos.current_price >= 0.90:
            _close(pos, pos.current_price, "TAKE_PROFIT")
        elif pos.current_price <= 0.10:
            _close(pos, pos.current_price, "STOP_LOSS")


def _close(pos: Position, price: float, reason: str):
    pos.pnl         = (price - pos.entry_price) * pos.shares
    pos.close_price = price
    pos.close_time  = datetime.now(timezone.utc).isoformat()
    pos.status      = "CLOSED"
    state.balance   += price * pos.shares
    state.total_pnl += pos.pnl
    state.closed_trades.append(asdict(pos))
    log_event(f"[CLOSE/{reason}] {pos.question[:45]} | PnL=${pos.pnl:+.2f} price={price:.3f}")


# ==============================================================================
#  STATE PERSISTENCE
# ==============================================================================
def save_state():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "balance":        state.balance,
            "total_pnl":      state.total_pnl,
            "scan_count":     state.scan_count,
            "signals_analyzed": state.signals_analyzed,
            "trades_taken":   state.trades_taken,
            "start_time":     state.start_time,
            "open_positions": [asdict(p) for p in state.positions if p.status == "OPEN"],
            "closed_trades":  state.closed_trades[-50:],
            "log":            state.log[-100:],
            "updated_at":     datetime.now(timezone.utc).isoformat(),
        }, f, indent=2, ensure_ascii=False)


def log_event(msg: str):
    ts    = datetime.now(timezone.utc).strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    state.log.append(entry)
    print(entry)


# ==============================================================================
#  MAIN LOOP
# ==============================================================================
async def scan_cycle():
    state.scan_count += 1
    log_event(f"-- Scan #{state.scan_count} " + "-" * 35)

    markets = await fetch_markets(50)
    update_positions(markets)

    open_ids   = {p.market_id for p in state.positions if p.status == "OPEN"}
    candidates = sorted(
        [m for m in markets if m.id not in open_ids and m.volume > 10_000],
        key=lambda m: m.volume, reverse=True
    )[:12]

    for market in candidates:
        state.signals_analyzed += 1

        arb    = detect_mispricing(market)
        signal = await analyse_market(market)

        if not signal:
            continue

        if arb:
            signal["edge"]       = max(signal["edge"], arb["gap"] * 0.5)
            signal["confidence"] = min(signal["confidence"] + 0.1, 0.95)

        allow, reason = risk_check(signal["edge"], signal["confidence"], market)
        if not allow:
            log_event(f"[SKIP] {market.question[:45]} -- {reason}")
            continue

        open_position(market, signal)
        await asyncio.sleep(1)

    save_state()
    log_event(
        f"Balance: ${state.balance:.2f} | "
        f"PnL: ${state.total_pnl:+.2f} | "
        f"Open: {sum(1 for p in state.positions if p.status == 'OPEN')}"
    )


async def main():
    log_event("Polymarket AI Paper Trading Bot started")
    log_event(f"LLM    : Replicate -> {REPLICATE_MODEL}")
    log_event(f"Balance: ${state.balance:.2f} USDC (PAPER)")
    log_event(f"Strategy: Llama 3.3 70B NLU + Arbitrage | Min edge: {MIN_EDGE_PCT:.0%}")
    log_event(f"Scan interval: {SCAN_INTERVAL}s | Max positions: {MAX_OPEN_POSITIONS}")
    log_event("-" * 60)

    while True:
        try:
            await scan_cycle()
        except KeyboardInterrupt:
            log_event("Bot stopped.")
            save_state()
            break
        except Exception as e:
            log_event(f"[LOOP ERROR] {e}")
        await asyncio.sleep(SCAN_INTERVAL)


import threading
from flask import Flask, send_file, jsonify

flask_app = Flask(__name__)

@flask_app.route("/")
def dashboard():
    return send_file("dashboard.html")

@flask_app.route("/paper_trades.json")
def trades():
    return send_file("paper_trades.json")

def run_flask():
    flask_app.run(host="0.0.0.0", port=8080)

# Start Flask in background thread
threading.Thread(target=run_flask, daemon=True).start()

if __name__ == "__main__":
    asyncio.run(main())