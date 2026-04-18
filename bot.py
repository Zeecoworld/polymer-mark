"""
Polymarket AI Trading Bot — v3 (Redis State Edition)
=====================================================
What changed from v2:
  - Redis replaces JSON file for ALL state persistence
  - total_pnl is NEVER lost on restart — restored from Redis on boot
  - Every trade open/close is written to Redis as an immutable ledger entry
  - Dashboard shows a PnL integrity check (Redis ledger vs running total)
  - Separate Redis keys for: state, open positions, closed trades, pnl ledger
  - Falls back to JSON file if Redis is unavailable (safe degraded mode)

REQUIRED ENV VARS (paper mode):
    REPLICATE_API_KEY      — get at replicate.com/account/api-tokens

REDIS (optional but strongly recommended):
    REDIS_URL              — e.g. redis://localhost:6379 or rediss://user:pass@host:6380
                             If not set, falls back to JSON file (v2 behaviour)

ADDITIONAL ENV VARS (live mode, LIVE_MODE=true):
    POLYMARKET_PRIVATE_KEY — wallet private key (no 0x prefix)
    POLYMARKET_API_KEY     — from polymarket.com → Profile → API Keys
    POLYMARKET_API_SECRET
    POLYMARKET_API_PASSPHRASE
    FUNDER_ADDRESS         — your Polygon wallet address
    SIGNATURE_TYPE         — 0=EOA/MetaMask, 1=Email/Magic (default 1)

OPTIONAL:
    TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID   — push alerts
    SLACK_WEBHOOK_URL                        — Slack alerts
    PAPER_BALANCE          — starting paper balance (default 1000)
    MAX_OPEN_POSITIONS     — default 3
    SCAN_INTERVAL          — seconds between scans (default 120)
    MIN_EDGE_PCT           — minimum edge to trade (default 0.05)
    MAX_POSITION_PCT       — max balance per trade (default 0.04)
    PORT                   — dashboard port (default 8080)
    LOG_FILE               — fallback JSON path (default paper_trades.json)

REDIS KEYS USED:
    polybot:state          — main state snapshot (balance, pnl, counters)
    polybot:positions      — list of open/closed Position objects
    polybot:closed_trades  — list of closed trade dicts (capped at 500)
    polybot:pnl_ledger     — append-only list of every PnL event (never deleted)
    polybot:log            — rolling bot log (last 200 entries)
"""

import asyncio
import json
import random
import httpx
import os
import xml.etree.ElementTree as ET
import aiofiles
import websockets
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from aiohttp import web as aio_web
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────────────────
LIVE_MODE              = os.getenv("LIVE_MODE", "false").lower() == "true"

# LLM
REPLICATE_API_KEY      = os.getenv("REPLICATE_API_KEY", "")
REPLICATE_MODEL        = os.getenv("REPLICATE_MODEL", "meta/meta-llama-3-70b-instruct")
REPLICATE_API_URL      = "https://api.replicate.com/v1/models/{model}/predictions"

# Wallet / CLOB
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
POLYMARKET_API_KEY_VAL = os.getenv("POLYMARKET_API_KEY", "")
POLYMARKET_API_SECRET  = os.getenv("POLYMARKET_API_SECRET", "")
POLYMARKET_API_PASS    = os.getenv("POLYMARKET_API_PASSPHRASE", "")
FUNDER_ADDRESS         = os.getenv("FUNDER_ADDRESS", "")
SIGNATURE_TYPE         = int(os.getenv("SIGNATURE_TYPE", "1"))

# Official Polymarket endpoints
CLOB_HOST              = "https://clob.polymarket.com"
GAMMA_API              = "https://gamma-api.polymarket.com"
WS_MARKET_URL          = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
WS_USER_URL            = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
CLOB_MIN_ORDER_SIZE    = 5.0

# Risk
PAPER_BALANCE_USDC     = float(os.getenv("PAPER_BALANCE", "1000.0"))
SCAN_INTERVAL          = int(os.getenv("SCAN_INTERVAL", "120"))
MAX_POSITION_PCT       = float(os.getenv("MAX_POSITION_PCT", "0.05"))
MIN_EDGE_PCT = float(os.getenv("MIN_EDGE_PCT", "0.02"))
MAX_OPEN_POSITIONS     = int(os.getenv("MAX_OPEN_POSITIONS", "3"))

# Alerts
TELEGRAM_TOKEN         = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID       = os.getenv("TELEGRAM_CHAT_ID", "")
SLACK_WEBHOOK          = os.getenv("SLACK_WEBHOOK_URL", "")

# Persistence
REDIS_URL              = os.getenv("REDIS_URL", "")          # empty = use JSON fallback
LOG_FILE               = os.getenv("LOG_FILE", "paper_trades.json")
TRADE_LOG_FILE         = "trade_history.jsonl"

# Redis key names
RK_STATE          = "polybot:state"
RK_POSITIONS      = "polybot:positions"
RK_CLOSED_TRADES  = "polybot:closed_trades"
RK_PNL_LEDGER     = "polybot:pnl_ledger"       # immutable — never trimmed
RK_LOG            = "polybot:log"

# ──────────────────────────────────────────────────────────────────────────────
#  STARTUP GUARDS
# ──────────────────────────────────────────────────────────────────────────────
if not REPLICATE_API_KEY:
    raise RuntimeError(
        "REPLICATE_API_KEY is not set.\n"
        "Get yours at https://replicate.com/account/api-tokens and set it as an env var."
    )

if LIVE_MODE:
    _missing = [v for v, val in [
        ("POLYMARKET_PRIVATE_KEY",    POLYMARKET_PRIVATE_KEY),
        ("POLYMARKET_API_KEY",        POLYMARKET_API_KEY_VAL),
        ("POLYMARKET_API_SECRET",     POLYMARKET_API_SECRET),
        ("POLYMARKET_API_PASSPHRASE", POLYMARKET_API_PASS),
        ("FUNDER_ADDRESS",            FUNDER_ADDRESS),
    ] if not val]
    if _missing:
        raise RuntimeError(f"LIVE_MODE=true but missing env vars: {', '.join(_missing)}")

# ──────────────────────────────────────────────────────────────────────────────
#  REDIS CLIENT  (lazy singleton)
# ──────────────────────────────────────────────────────────────────────────────
_redis = None
_redis_available = False


async def get_redis():
    """
    Returns async Redis client, or None if Redis is unavailable.
    Falls back gracefully — bot continues with JSON if Redis is down.
    """
    global _redis, _redis_available
    if _redis is not None:
        return _redis if _redis_available else None
    if not REDIS_URL:
        return None
    try:
        import redis.asyncio as aioredis
        _redis = aioredis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
            retry_on_timeout=True,
        )
        await _redis.ping()
        _redis_available = True
        print(f"[REDIS] Connected to {REDIS_URL[:30]}…")
        return _redis
    except ImportError:
        print("[REDIS] redis package not installed — run: pip install redis")
        return None
    except Exception as e:
        print(f"[REDIS] Connection failed: {e} — falling back to JSON file")
        _redis_available = False
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  REDIS STATE OPERATIONS
# ──────────────────────────────────────────────────────────────────────────────
async def redis_save_state(state_obj) -> bool:
    """Save all mutable state to Redis atomically using a pipeline."""
    r = await get_redis()
    if r is None:
        return False
    try:
        pipe = r.pipeline()

        # Main state snapshot
        pipe.set(RK_STATE, json.dumps({
            "balance":          state_obj.balance,
            "total_pnl":        state_obj.total_pnl,
            "scan_count":       state_obj.scan_count,
            "signals_analyzed": state_obj.signals_analyzed,
            "trades_taken":     state_obj.trades_taken,
            "start_time":       state_obj.start_time,
            "mode":             "LIVE" if LIVE_MODE else "PAPER",
            "updated_at":       datetime.now(timezone.utc).isoformat(),
        }))

        # Open + recent closed positions
        pipe.set(RK_POSITIONS, json.dumps(
            [asdict(p) for p in state_obj.positions]
        ))

        # Closed trades list (keep last 500 in Redis)
        pipe.set(RK_CLOSED_TRADES, json.dumps(
            state_obj.closed_trades[-500:]
        ))

        # Rolling log
        pipe.set(RK_LOG, json.dumps(state_obj.log[-200:]))

        await pipe.execute()
        return True
    except Exception as e:
        print(f"[REDIS] save_state failed: {e}")
        return False


async def redis_load_state() -> Optional[dict]:
    """Load all state from Redis. Returns None if unavailable."""
    r = await get_redis()
    if r is None:
        return None
    try:
        pipe = r.pipeline()
        pipe.get(RK_STATE)
        pipe.get(RK_POSITIONS)
        pipe.get(RK_CLOSED_TRADES)
        pipe.get(RK_LOG)
        results = await pipe.execute()

        if not results[0]:
            return None   # no state saved yet

        state_data    = json.loads(results[0])
        positions     = json.loads(results[1]) if results[1] else []
        closed_trades = json.loads(results[2]) if results[2] else []
        log_entries   = json.loads(results[3]) if results[3] else []

        return {
            **state_data,
            "positions":     positions,
            "closed_trades": closed_trades,
            "log":           log_entries,
        }
    except Exception as e:
        print(f"[REDIS] load_state failed: {e}")
        return None


async def redis_append_pnl_ledger(entry: dict):
    """
    Append an immutable PnL ledger entry to Redis.
    This list is NEVER trimmed — it's the source of truth for all-time PnL.
    Used to verify total_pnl is correct and detect any accounting drift.
    """
    r = await get_redis()
    if r is None:
        return
    try:
        await r.rpush(RK_PNL_LEDGER, json.dumps({
            **entry,
            "ts": datetime.now(timezone.utc).isoformat(),
        }))
    except Exception as e:
        print(f"[REDIS] append_pnl_ledger failed: {e}")


async def redis_get_ledger_total() -> Optional[float]:
    """
    Sum all PnL entries from the immutable ledger.
    Used by the dashboard integrity check to verify total_pnl is correct.
    """
    r = await get_redis()
    if r is None:
        return None
    try:
        entries = await r.lrange(RK_PNL_LEDGER, 0, -1)
        total = sum(float(json.loads(e).get("pnl", 0)) for e in entries)
        return round(total, 4)
    except Exception as e:
        print(f"[REDIS] get_ledger_total failed: {e}")
        return None


async def redis_get_ledger_entries(limit: int = 100) -> list:
    """Fetch most recent ledger entries for the dashboard."""
    r = await get_redis()
    if r is None:
        return []
    try:
        entries = await r.lrange(RK_PNL_LEDGER, -limit, -1)
        return [json.loads(e) for e in reversed(entries)]
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
#  CLOB CLIENT  (lazy singleton, only used in LIVE_MODE)
# ──────────────────────────────────────────────────────────────────────────────
_clob_client = None

def get_clob_client():
    global _clob_client
    if _clob_client is not None:
        return _clob_client
    try:
        from py_clob_client.client import ClobClient
    except ImportError:
        raise RuntimeError("py-clob-client not installed.\nRun: pip install py-clob-client")
    _clob_client = ClobClient(
        host=CLOB_HOST, chain_id=137,
        key=POLYMARKET_PRIVATE_KEY,
        signature_type=SIGNATURE_TYPE,
        funder=FUNDER_ADDRESS,
    )
    _clob_client.set_api_creds(_clob_client.create_or_derive_api_creds())
    return _clob_client


async def clob_place_limit_order(token_id: str, size_usdc: float, price: float) -> Optional[dict]:
    try:
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY
        limit_price = round(min(price + 0.01, 0.99), 3)
        size_shares = round(size_usdc / limit_price, 4)
        def _place():
            client = get_clob_client()
            order = client.create_order(OrderArgs(token_id=token_id, price=limit_price, size=size_shares, side=BUY))
            return client.post_order(order, OrderType.GTC)
        result = await asyncio.to_thread(_place)
        log_event(f"[CLOB ORDER] token={token_id[:14]}… price={limit_price} shares={size_shares} status={result.get('status','?')}")
        return result
    except Exception as e:
        log_event(f"[CLOB ERROR] Limit order failed: {e}")
        return None


async def clob_cancel_order(order_id: str) -> bool:
    try:
        await asyncio.to_thread(lambda: get_clob_client().cancel(order_id))
        log_event(f"[CLOB] Cancelled {order_id}")
        return True
    except Exception as e:
        log_event(f"[CLOB] Cancel failed {order_id}: {e}")
        return False


async def clob_get_balance() -> float:
    try:
        raw = await asyncio.to_thread(lambda: get_clob_client().get_balance())
        return float(raw.get("balance", 0))
    except Exception as e:
        log_event(f"[CLOB] get_balance failed: {e}")
        return 0.0


async def clob_get_midpoint(token_id: str) -> Optional[float]:
    def _fetch():
        from py_clob_client.client import ClobClient
        return ClobClient(CLOB_HOST).get_midpoint(token_id)
    try:
        result = await asyncio.to_thread(_fetch)
        if result and "mid" in result:
            return float(result["mid"])
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  WEBSOCKET PRICE FEED
# ──────────────────────────────────────────────────────────────────────────────
_ws_prices: dict[str, float] = {}
_ws_subscribed: set[str] = set()


async def ws_price_feed():
    while True:
        try:
            tracked = list({p.token_id for p in state.positions if p.status == "OPEN" and p.token_id})
            if not tracked:
                await asyncio.sleep(15)
                continue
            log_event(f"[WS] Connecting — tracking {len(tracked)} token(s)")
            async with websockets.connect(WS_MARKET_URL, ping_interval=None, close_timeout=5, open_timeout=10) as ws:
                await ws.send(json.dumps({"assets_ids": tracked, "type": "market", "custom_feature_enabled": True}))
                _ws_subscribed.update(tracked)
                log_event(f"[WS] Subscribed to {len(tracked)} token(s)")
                last_ping = asyncio.get_event_loop().time()
                async for raw in ws:
                    now = asyncio.get_event_loop().time()
                    if now - last_ping >= 10:
                        await ws.send(json.dumps({"type": "PING"}))
                        last_ping = now
                    new_tokens = {p.token_id for p in state.positions if p.status == "OPEN" and p.token_id} - _ws_subscribed
                    if new_tokens:
                        await ws.send(json.dumps({"assets_ids": list(new_tokens), "type": "market", "custom_feature_enabled": True}))
                        _ws_subscribed.update(new_tokens)
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue

                    if isinstance(msg, list):
                        msg = msg[0] if msg else {}

                    event_type = msg.get("event_type", "")
                    asset_id   = msg.get("asset_id", "")
                    if event_type == "last_trade_price" and asset_id:
                        price = msg.get("price")
                        if price:
                            _ws_prices[asset_id] = float(price)
                    elif event_type == "price_change" and asset_id:
                        for change in msg.get("price_changes", []):
                            bid, ask = change.get("best_bid"), change.get("best_ask")
                            if bid and ask:
                                _ws_prices[asset_id] = (float(bid) + float(ask)) / 2
                    elif event_type == "book" and asset_id:
                        bids, asks = msg.get("bids", []), msg.get("asks", [])
                        if bids and asks:
                            _ws_prices[asset_id] = (float(bids[0]["price"]) + float(asks[0]["price"])) / 2
        except Exception as e:
            log_event(f"[WS] Disconnected: {e} — reconnecting in 15s")
            await asyncio.sleep(15)


def get_ws_price(token_id: str) -> Optional[float]:
    return _ws_prices.get(token_id)


# ──────────────────────────────────────────────────────────────────────────────
#  RSS NEWS
# ──────────────────────────────────────────────────────────────────────────────
RSS_FEEDS = {
    "crypto":  [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://news.google.com/rss/search?q=bitcoin+crypto&hl=en",
    ],
    "general": [
        "https://feeds.reuters.com/reuters/topNews",
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://news.google.com/rss/headlines?hl=en",
    ],
    "tech": [
        "https://news.google.com/rss/search?q=AI+technology+announcement&hl=en",
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
    ],
}

_news_cache: dict = {}
_news_cache_time: dict = {}
NEWS_CACHE_TTL = 300
_http: Optional[httpx.AsyncClient] = None


async def get_http() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(
            timeout=15, follow_redirects=True,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=4),
        )
    return _http


async def fetch_rss(url: str) -> list[str]:
    try:
        http = await get_http()
        r = await http.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.text)
        headlines = []
        for item in root.iter("item"):
            t = item.find("title")
            if t is not None and t.text:
                headlines.append(t.text.strip())
        for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
            t = entry.find("{http://www.w3.org/2005/Atom}title")
            if t is not None and t.text:
                headlines.append(t.text.strip())
        return headlines[:10]
    except Exception:
        return []


async def get_news_for_market(question: str, category: str) -> str:
    now = datetime.now(timezone.utc).timestamp()
    q = question.lower()
    if any(w in q for w in ["bitcoin","btc","eth","crypto","solana","coinbase","binance","blockchain","defi","nft","altcoin"]):
        feed_keys = ["crypto", "general"]
    elif any(w in q for w in ["gpt","ai","openai","apple","google","microsoft","meta","launch","release","announce","model"]):
        feed_keys = ["tech", "general"]
    else:
        feed_keys = ["general"]

    all_headlines: list[str] = []
    for key in feed_keys[:2]:
        for url in RSS_FEEDS.get(key, [])[:2]:
            if url in _news_cache and now - _news_cache_time.get(url, 0) < NEWS_CACHE_TTL:
                all_headlines.extend(_news_cache[url])
            else:
                h = await fetch_rss(url)
                _news_cache[url] = h
                _news_cache_time[url] = now
                all_headlines.extend(h)

    if not all_headlines:
        return "No recent news found."
    keywords = [w for w in question.replace("?","").lower().split() if len(w) > 3]
    def relevance(h: str) -> int:
        return sum(1 for kw in keywords if kw in h.lower())
    scored = sorted(set(all_headlines), key=relevance, reverse=True)
    top = [h for h in scored if relevance(h) > 0][:5] or scored[:3]
    return " | ".join(top) if top else "No relevant news found."


# ──────────────────────────────────────────────────────────────────────────────
#  ALERTS
# ──────────────────────────────────────────────────────────────────────────────
async def send_alert(msg: str):
    tasks = []
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        tasks.append(_telegram_alert(msg))
    if SLACK_WEBHOOK:
        tasks.append(_slack_alert(msg))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

async def _telegram_alert(msg: str):
    try:
        http = await get_http()
        await http.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": f"🤖 PolyBot\n{msg}", "parse_mode": "Markdown"})
    except Exception as e:
        log_event(f"[ALERT] Telegram failed: {e}")

async def _slack_alert(msg: str):
    try:
        http = await get_http()
        await http.post(SLACK_WEBHOOK, json={"text": f"🤖 PolyBot: {msg}"})
    except Exception as e:
        log_event(f"[ALERT] Slack failed: {e}")



@dataclass
class Market:
    id:           str
    condition_id: str
    question:     str
    yes_price:    float
    no_price:     float
    volume:       float
    end_date:     str
    category:     str = "misc"
    description:  str = ""
    yes_token_id: str = ""
    no_token_id:  str = ""


@dataclass
class Position:
    market_id:         str
    condition_id:      str
    question:          str
    side:              str
    shares:            float
    entry_price:       float
    current_price:     float
    timestamp:         str
    claude_confidence: float
    edge:              float
    status:            str   = "OPEN"
    pnl:               float = 0.0
    close_price:       float = 0.0
    close_time:        str   = ""
    order_id:          str   = ""
    token_id:          str   = ""
    source:            str   = "ai"


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



def load_saved_state():
    """
    Restore full state on startup.
    Tries Redis first, falls back to JSON file.
    This is what prevents total_pnl resetting to 0 on restart.
    """
    # We need to run async Redis load in a sync context at startup
    loop = asyncio.get_event_loop()

    async def _try_redis():
        return await redis_load_state()

    try:
        saved = loop.run_until_complete(_try_redis())
    except Exception:
        saved = None

    # Fall back to JSON file if Redis had nothing
    if saved is None and os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            print(f"[STATE] Loaded from JSON fallback: {LOG_FILE}")
        except Exception as e:
            print(f"[STATE] JSON load failed: {e}")
            saved = None

    if saved is None:
        print("[STATE] No saved state found — starting fresh")
        return

    # Restore scalar fields
    state.balance          = float(saved.get("balance",          PAPER_BALANCE_USDC))
    state.total_pnl        = float(saved.get("total_pnl",        0.0))
    state.scan_count       = int(saved.get("scan_count",         0))
    state.signals_analyzed = int(saved.get("signals_analyzed",   0))
    state.trades_taken     = int(saved.get("trades_taken",       0))
    state.start_time       = saved.get("start_time",             state.start_time)
    state.closed_trades    = saved.get("closed_trades",          [])
    state.log              = saved.get("log",                    [])

    # Restore Position objects from saved dicts
    raw_positions = saved.get("positions") or saved.get("open_positions") or []
    restored = []
    for p in raw_positions:
        try:
            restored.append(Position(**{k: p[k] for k in Position.__dataclass_fields__ if k in p}))
        except Exception:
            pass
    state.positions = restored

    source = "Redis" if _redis_available else "JSON"
    print(
        f"[STATE] Restored from {source} — "
        f"balance=${state.balance:.2f} "
        f"total_pnl=${state.total_pnl:+.2f} "
        f"trades={state.trades_taken} "
        f"open_positions={len([p for p in state.positions if p.status == 'OPEN'])}"
    )


def _write_json_fallback():
    """Write JSON snapshot — used as fallback when Redis is unavailable."""
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "balance":          state.balance,
                "total_pnl":        state.total_pnl,
                "scan_count":       state.scan_count,
                "signals_analyzed": state.signals_analyzed,
                "trades_taken":     state.trades_taken,
                "start_time":       state.start_time,
                "mode":             "LIVE" if LIVE_MODE else "PAPER",
                "positions":        [asdict(p) for p in state.positions],
                "open_positions":   [asdict(p) for p in state.positions if p.status == "OPEN"],
                "closed_trades":    state.closed_trades[-100:],
                "log":              state.log[-200:],
                "updated_at":       datetime.now(timezone.utc).isoformat(),
            }, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[JSON] Write failed: {e}")


async def save_state():
    """Save to Redis (primary) and JSON file (fallback) simultaneously."""
    redis_ok = await redis_save_state(state)
    # Always write JSON too — belt and suspenders
    await asyncio.to_thread(_write_json_fallback)
    return redis_ok


async def append_trade_log(trade: dict):
    """Write to both append-only JSONL file AND Redis PnL ledger."""
    # 1. Local JSONL file (always)
    async with aiofiles.open(TRADE_LOG_FILE, "a", encoding="utf-8") as f:
        await f.write(json.dumps(trade, ensure_ascii=False) + "\n")

    # 2. Redis PnL ledger — only for close events (these define profit/loss)
    if trade.get("event") == "close":
        await redis_append_pnl_ledger({
            "trade_id":    trade.get("condition_id", ""),
            "question":    trade.get("question", "")[:60],
            "side":        trade.get("side", ""),
            "pnl":         trade.get("pnl", 0),
            "entry_price": trade.get("entry_price", 0),
            "close_price": trade.get("close_price", 0),
            "shares":      trade.get("shares", 0),
            "reason":      trade.get("reason", ""),
            "mode":        trade.get("mode", "PAPER"),
        })


def log_event(msg: str):
    ts    = datetime.now(timezone.utc).strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    state.log.append(entry)
    if len(state.log) > 200:
        state.log = state.log[-200:]
    print(entry)


# ──────────────────────────────────────────────────────────────────────────────
#  MEMORY PRUNING
# ──────────────────────────────────────────────────────────────────────────────
def prune_memory():
    if len(state.log) > 200:
        state.log = state.log[-200:]
    open_pos   = [p for p in state.positions if p.status == "OPEN"]
    closed_pos = [p for p in state.positions if p.status == "CLOSED"][-20:]
    state.positions = open_pos + closed_pos
    if len(state.closed_trades) > 500:
        state.closed_trades = state.closed_trades[-500:]
    now   = datetime.now(timezone.utc).timestamp()
    stale = [k for k, t in _news_cache_time.items() if now - t > NEWS_CACHE_TTL * 2]
    for k in stale:
        _news_cache.pop(k, None)
        _news_cache_time.pop(k, None)


# ──────────────────────────────────────────────────────────────────────────────
#  GAMMA API — MARKET FETCHING
# ──────────────────────────────────────────────────────────────────────────────
async def fetch_markets(limit: int = 50) -> list[Market]:
    try:
        http = await get_http()
        r = await http.get(f"{GAMMA_API}/markets", params={
            "active": "true", "closed": "false",
            "enableOrderBook": "true", "acceptingOrders": "true",
            "limit": limit, "order": "volumeNum", "ascending": "false",
        })
        r.raise_for_status()
        items = r.json()
        if not isinstance(items, list):
            items = items.get("markets", [])
        markets: list[Market] = []
        for m in items:
            try:
                op  = m.get("outcomePrices")
                out = m.get("outcomes", '["Yes","No"]')
                if not op:
                    continue
                prices   = json.loads(op)  if isinstance(op, str) else op
                outcomes = json.loads(out) if isinstance(out, str) else out
                yi = next((i for i, o in enumerate(outcomes) if o.lower() in ("yes","true")), 0)
                ni = next((i for i, o in enumerate(outcomes) if o.lower() in ("no","false")), 1)
                yes_p, no_p = float(prices[yi]), float(prices[ni])
                if yes_p <= 0 or no_p <= 0:
                    continue
                yes_token = no_token = ""
                try:
                    clob_ids = json.loads(m.get("clobTokenIds","[]"))
                    if len(clob_ids) >= 2:
                        yes_token, no_token = clob_ids[yi], clob_ids[ni]
                except Exception:
                    pass
                markets.append(Market(
                    id=str(m.get("id","")),
                    condition_id=m.get("conditionId", str(m.get("id",""))),
                    question=m.get("question",""),
                    yes_price=yes_p, no_price=no_p,
                    volume=float(m.get("volumeNum") or m.get("volume24hr") or 0),
                    end_date=m.get("endDate",""),
                    category=m.get("category") or "misc",
                    description=(m.get("description") or "")[:400],
                    yes_token_id=yes_token, no_token_id=no_token,
                ))
            except Exception:
                continue
        log_event(f"[API] Fetched {len(markets)} CLOB-tradeable markets")
        return markets
    except Exception as e:
        log_event(f"[API ERROR] fetch_markets: {e} — using mock data")
        return _mock_markets()


def _mock_markets() -> list[Market]:
    base = [
        ("Will BTC close above $90k on Apr 30?",  0.42, 0.58, 2_100_000, "crypto"),
        ("Will ETH hit $4k before May 1?",         0.31, 0.69,   890_000, "crypto"),
        ("Will Fed cut rates in May 2026?",        0.38, 0.62, 1_200_000, "finance"),
        ("Will GPT-5 be announced before June?",  0.44, 0.56,   410_000, "tech"),
        ("Will NVDA stock be above $900 by May?", 0.61, 0.39,   560_000, "finance"),
        ("Will Iran nuclear deal be signed?",      0.22, 0.78,   320_000, "politics"),
        ("Will US enter recession by Q3 2026?",   0.35, 0.65,   780_000, "finance"),
        ("Will OpenAI release new model in May?", 0.58, 0.42,   290_000, "tech"),
    ]
    now = datetime.now(timezone.utc)
    return [
        Market(id=f"mock_{i}", condition_id=f"mock_cid_{i}", question=q,
               yes_price=round(yp+random.uniform(-0.02,0.02),3),
               no_price=round(np_+random.uniform(-0.02,0.02),3),
               volume=vol, end_date=(now+timedelta(days=random.randint(3,30))).isoformat(), category=cat)
        for i,(q,yp,np_,vol,cat) in enumerate(base)
    ]


async def fetch_single_market_price(market_id: str, side: str) -> Optional[float]:
    try:
        http = await get_http()
        r = await http.get(f"{GAMMA_API}/markets/{market_id}")
        r.raise_for_status()
        m = r.json()
        op  = m.get("outcomePrices")
        out = m.get("outcomes", '["Yes","No"]')
        if op:
            prices   = json.loads(op)  if isinstance(op, str) else op
            outcomes = json.loads(out) if isinstance(out, str) else out
            if side == "YES":
                yi = next((i for i,o in enumerate(outcomes) if o.lower() in ("yes","true")), 0)
                return float(prices[yi])
            else:
                ni = next((i for i,o in enumerate(outcomes) if o.lower() in ("no","false")), 1)
                return float(prices[ni])
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  PRICE MOMENTUM SIGNAL
# ──────────────────────────────────────────────────────────────────────────────
_price_history: dict[str, list] = {}


def record_price(condition_id: str, price: float):
    hist = _price_history.setdefault(condition_id, [])
    hist.append(price)
    if len(hist) > 10:
        hist.pop(0)


def get_momentum_signal(condition_id: str, current_price: float) -> Optional[float]:
    hist = _price_history.get(condition_id, [])
    if len(hist) < 3:
        return None
    n = len(hist)
    x_mean, y_mean = (n-1)/2, sum(hist)/n
    num = sum((i-x_mean)*(hist[i]-y_mean) for i in range(n))
    den = sum((i-x_mean)**2 for i in range(n))
    if den == 0:
        return None
    slope = num / den
    if abs(slope) < 0.01:
        return None
    return round(max(0.02, min(0.98, current_price + slope * 3)), 4)


# ──────────────────────────────────────────────────────────────────────────────
#  LLM ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
async def call_replicate_llama(prompt: str) -> str:
    system = "You are a quantitative prediction market analyst. Always respond with valid JSON only. No markdown fences, no extra text."
    formatted = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    headers = {"Authorization": f"Bearer {REPLICATE_API_KEY}", "Content-Type": "application/json", "Prefer": "wait"}
    payload = {"input": {"prompt": formatted, "max_tokens": 300, "temperature": 0.2, "stop": ["<|eot_id|>", "<|end_of_text|>"]}}
    url = REPLICATE_API_URL.format(model=REPLICATE_MODEL)
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as http:
        r = await http.post(url, headers=headers, json=payload)
        r.raise_for_status()
        result = r.json()
        output = result.get("output")
        if isinstance(output, list): return "".join(output).strip()
        if isinstance(output, str):  return output.strip()
        poll_url = result.get("urls", {}).get("get", "")
        if not poll_url:
            raise ValueError(f"No output and no poll URL: {result}")
        for _ in range(30):
            await asyncio.sleep(1)
            poll = await http.get(poll_url, headers=headers)
            poll.raise_for_status()
            data = poll.json()
            if data.get("status") == "succeeded":
                out = data.get("output", [])
                return ("".join(out) if isinstance(out, list) else out).strip()
            if data.get("status") in ("failed", "canceled"):
                raise ValueError(f"Replicate job failed: {data.get('error')}")
        raise TimeoutError("Replicate timed out after 30s")


async def analyse_market_llm_only(market: Market) -> Optional[dict]:
    news = await get_news_for_market(market.question, market.category)
    days_left = "unknown"
    try:
        end = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
        days_left = str(max(0, (end - datetime.now(timezone.utc)).days))
    except Exception:
        pass

    prompt = (
        f"You are a prediction market trader looking for mispriced markets.\n\n"
        f"Question: {market.question}\n"
        f"Category: {market.category}\n"
        f"Days until resolution: {days_left}\n"
        f"Market YES price: {market.yes_price:.3f} ({market.yes_price*100:.1f}% implied)\n"
        f"Market NO price:  {market.no_price:.3f}\n"
        f"Volume: ${market.volume:,.0f}\n"
        f"Description: {market.description or 'N/A'}\n"
        f"Recent news: {news}\n\n"
        "Your job: estimate the TRUE probability of YES resolving.\n"
        "Be willing to disagree with the market if you have reason to.\n"
        "A confidence of 0.60 means you are moderately sure of your estimate.\n"
        "A confidence of 0.50 means you have some basis but limited news.\n"
        "Never set confidence below 0.45 unless you have no information at all.\n\n"
        "Respond ONLY with this exact JSON, no other text:\n"
        '{"true_prob_yes": 0.55, "confidence": 0.60, "reasoning": "one sentence"}\n'
    )

    try:
        text = await call_replicate_llama(prompt)
        log_event(f"[LLM RAW] {text[:100]}")
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        s, e = text.find("{"), text.rfind("}") + 1
        if s == -1 or e <= s:
            return None
        data = json.loads(text[s:e])

        true_prob = float(data.get("true_prob_yes", market.yes_price))
        conf      = float(data.get("confidence", 0.50))

        # Single soft guard — if LLM gives nonsense extremes, pull back slightly
        if true_prob < 0.03 or true_prob > 0.97:
            true_prob = market.yes_price  # no edge at extremes, use market price
            conf = 0.45

        true_prob = max(0.01, min(0.99, true_prob))
        conf      = max(0.40, min(0.99, conf))  # floor at 0.40, never lower

        return {
            "true_prob":  true_prob,
            "confidence": conf,
            "reasoning":  data.get("reasoning", ""),
            "source":     "llm",
        }
    except Exception as e:
        log_event(f"[LLM ERROR] {market.question[:45]}: {e}")
        log_event(f"[LLM RAW] {text[:120] if 'text' in dir() else 'no response'}")
        return None


def detect_mispricing(market: Market) -> Optional[dict]:
    spread = market.yes_price + market.no_price
    if spread < 0.94:
        gap  = 1.0 - spread
        side = "YES" if market.yes_price <= market.no_price else "NO"
        return {"type": "spread_arb", "gap": gap, "side": side,
                "token_id": market.yes_token_id if side == "YES" else market.no_token_id}
    return None


async def analyse_market(market: Market) -> Optional[dict]:
    record_price(market.condition_id, market.yes_price)
    signals = []

    llm_result = await analyse_market_llm_only(market)
    if llm_result and llm_result["confidence"] >= 0.30:  # was 0.45 — too high after penalties
        edge_yes = llm_result["true_prob"] - market.yes_price
        edge_no  = (1 - llm_result["true_prob"]) - market.no_price
        side = "YES" if edge_yes >= edge_no else "NO"
        edge = edge_yes if side == "YES" else edge_no
        if abs(edge) > 0.02:   # was 0.04 — lowered to catch more real edges
            signals.append({"source":"llm","side":side,"edge":edge,
                            "confidence":llm_result["confidence"],"weight":0.50,
                            "reasoning":llm_result.get("reasoning","")})

    mom_prob = get_momentum_signal(market.condition_id, market.yes_price)
    if mom_prob is not None:
        edge_yes = mom_prob - market.yes_price
        edge_no  = (1 - mom_prob) - market.no_price
        side = "YES" if edge_yes >= edge_no else "NO"
        edge = edge_yes if side == "YES" else edge_no
        if abs(edge) > 0.04:
            signals.append({"source":"momentum","side":side,"edge":edge,"confidence":0.55,"weight":0.35})

    arb = detect_mispricing(market)
    if arb:
        signals.append({"source":"arb","side":arb["side"],"edge":arb["gap"]*0.6,"confidence":0.80,"weight":0.15})

    if not signals: return None

    yes_sigs = [s for s in signals if s["side"] == "YES"]
    no_sigs  = [s for s in signals if s["side"] == "NO"]
    dominant = yes_sigs if len(yes_sigs) >= len(no_sigs) else no_sigs
    side = "YES" if dominant is yes_sigs else "NO"

    # ── Signal agreement: relaxed for paper trading ──
    # Old rule: needed 2+ signals OR conf ≥ 0.70
    # Problem: momentum needs 3+ scans to warm up, arb is rare on liquid markets
    # So LLM was the only signal but 1 signal always got rejected
    # New rule: 1 strong LLM signal (conf ≥ 0.52) is enough to proceed
    # 2+ signals still get a confidence boost (reward for agreement)
    if len(dominant) == 0:
        return None

    if len(dominant) == 1:
        solo = dominant[0]
        if solo["confidence"] < 0.32:
            log_event(f"[SKIP] Solo signal conf={solo['confidence']:.2f} < 0.52 for {market.question[:40]}")
            return None
        # Solo signal passes — log it clearly
        log_event(f"[SIGNAL] Solo {solo['source']} signal accepted conf={solo['confidence']:.2f}")
    else:
        # Multi-signal agreement — boost confidence by 8%
        for s in dominant:
            s["confidence"] = min(s["confidence"] + 0.08, 0.95)
        log_event(f"[SIGNAL] {len(dominant)}-signal agreement — confidence boosted")

    total_w  = sum(s["weight"] for s in dominant)
    avg_conf = sum(s["confidence"]*s["weight"] for s in dominant) / total_w
    avg_edge = sum(s["edge"]*s["weight"]       for s in dominant) / total_w

    # Composite gate — lowered from 0.04 to 0.025 for paper mode
    # 0.04 was too high: e.g. conf=0.55 × edge=0.06 = 0.033 → rejected
    if avg_conf * avg_edge < 0.008:
        log_event(f"[SKIP] Composite {avg_conf*avg_edge:.4f} too low (need ≥ 0.025) for {market.question[:40]}")
        return None

    sources  = "+".join(sorted({s["source"] for s in dominant}))
    token_id = market.yes_token_id if side == "YES" else market.no_token_id
    entry    = market.yes_price    if side == "YES" else market.no_price
    reasoning = next((s.get("reasoning","") for s in dominant if s.get("reasoning")), "")
    log_event(f"[SIGNAL ✓] {side} {market.question[:40]} | edge={avg_edge:.3f} conf={avg_conf:.2f} src={sources}")
    return {"side":side,"entry_price":entry,"token_id":token_id,
            "edge":round(avg_edge,4),"confidence":round(avg_conf,4),
            "reasoning":reasoning,"source":sources}


# ──────────────────────────────────────────────────────────────────────────────
#  RISK MANAGER
# ──────────────────────────────────────────────────────────────────────────────
def risk_check(edge: float, confidence: float, market: Market) -> tuple[bool, str]:
    open_pos = [p for p in state.positions if p.status == "OPEN"]
    if len(open_pos) >= MAX_OPEN_POSITIONS:  return False, f"Max positions ({MAX_OPEN_POSITIONS}) reached"
    if edge < MIN_EDGE_PCT:                  return False, f"Edge {edge:.3f} below min {MIN_EDGE_PCT}"
    # Lowered from 0.50 → 0.45 — calibration penalties already pushed conf down
    # 0.50 was rejecting every trade that had even one penalty applied
    if confidence < 0.32:                    return False, f"Confidence {confidence:.2f} too low (need ≥ 0.45)"
    if market.volume < 10_000: return False, f"Volume ${market.volume:,.0f} below $10k floor"
    if market.yes_price < 0.05 or market.yes_price > 0.95: return False, f"Price {market.yes_price:.3f} too extreme"
    if market.condition_id in {p.condition_id for p in open_pos}: return False, "Already have position"
    if LIVE_MODE:
        sz = position_size(edge, confidence)
        if sz < CLOB_MIN_ORDER_SIZE: return False, f"Size ${sz:.2f} below CLOB minimum"
    return True, "OK"


def position_size(edge: float, confidence: float) -> float:
    kelly = (edge * confidence) / max(1 - (edge * confidence), 0.01)
    return round(min(state.balance * kelly * 0.25, state.balance * MAX_POSITION_PCT, state.balance * 0.05), 2)


# ──────────────────────────────────────────────────────────────────────────────
#  POSITION EXECUTION
# ──────────────────────────────────────────────────────────────────────────────
async def open_position(market: Market, signal: dict) -> Optional[Position]:
    size = position_size(signal["edge"], signal["confidence"])
    if size < 1.0 or size > state.balance:
        log_event(f"[SKIP] Size ${size:.2f} invalid for {market.question[:40]}")
        return None

    order_id = ""
    token_id = signal.get("token_id", "")

    if LIVE_MODE:
        if not token_id:
            log_event(f"[LIVE SKIP] No token_id for {market.question[:40]}")
            return None
        result = await clob_place_limit_order(token_id, size, signal["entry_price"])
        if not result:
            log_event(f"[LIVE FAIL] Order not placed for {market.question[:40]}")
            return None
        order_id = result.get("orderID") or result.get("id") or ""
        live_bal = await clob_get_balance()
        state.balance = live_bal if live_bal > 0 else state.balance - size
        await send_alert(
            f"🤖 LIVE TRADE OPENED\n{market.question[:50]}\n"
            f"{signal['side']} @ {signal['entry_price']:.3f} | edge={signal['edge']:.3f} | ${size:.2f}\n"
            f"Signals: {signal.get('source','')} | {signal.get('reasoning','')}"
        )
    else:
        state.balance -= size

    shares = round(size / signal["entry_price"], 4)
    pos = Position(
        market_id=market.id, condition_id=market.condition_id, question=market.question,
        side=signal["side"], shares=shares, entry_price=signal["entry_price"],
        current_price=signal["entry_price"], timestamp=datetime.now(timezone.utc).isoformat(),
        claude_confidence=signal["confidence"], edge=signal["edge"],
        order_id=order_id, token_id=token_id, source=signal.get("source","ai"),
    )
    state.positions.append(pos)
    state.trades_taken += 1

    mode = "LIVE" if LIVE_MODE else "PAPER"
    log_event(f"[{mode} OPEN] {signal['side']} {market.question[:50]} | edge={signal['edge']:.3f} conf={signal['confidence']:.2f} src={signal.get('source','?')} size=${size:.2f}")
    await append_trade_log({**asdict(pos), "event": "open", "mode": mode, "ts": datetime.now(timezone.utc).isoformat()})
    return pos


# ──────────────────────────────────────────────────────────────────────────────
#  POSITION UPDATES & CLOSING
# ──────────────────────────────────────────────────────────────────────────────
async def update_positions(markets: list[Market]):
    market_lookup = {m.condition_id: m for m in markets}
    market_lookup.update({m.id: m for m in markets})

    for pos in state.positions:
        if pos.status != "OPEN":
            continue
        new_price = None
        if pos.token_id:
            new_price = get_ws_price(pos.token_id)
        if new_price is None and pos.token_id and not pos.market_id.startswith("mock_"):
            new_price = await clob_get_midpoint(pos.token_id)
        if new_price is None:
            live = market_lookup.get(pos.condition_id) or market_lookup.get(pos.market_id)
            if live:
                new_price = live.yes_price if pos.side == "YES" else live.no_price
        if new_price is None and not pos.market_id.startswith("mock_"):
            new_price = await fetch_single_market_price(pos.market_id, pos.side)
        if new_price is not None:
            pos.current_price = new_price
        pos.pnl = round((pos.current_price - pos.entry_price) * pos.shares, 4)

        take_profit = min(pos.entry_price + 0.30, 0.88)
        stop_loss   = max(pos.entry_price - 0.25, 0.08)
        if pos.current_price >= take_profit:
            await _close_position(pos, pos.current_price, "TAKE_PROFIT")
        elif pos.current_price <= stop_loss:
            await _close_position(pos, pos.current_price, "STOP_LOSS")


async def _close_position(pos: Position, price: float, reason: str):
    pos.pnl         = round((price - pos.entry_price) * pos.shares, 4)
    pos.close_price = price
    pos.close_time  = datetime.now(timezone.utc).isoformat()
    pos.status      = "CLOSED"
    state.balance   = round(state.balance + price * pos.shares, 2)
    # ── Incrementally maintain total_pnl — NEVER recalculate from list ──
    state.total_pnl = round(state.total_pnl + pos.pnl, 4)
    state.closed_trades.append(asdict(pos))

    mode = "LIVE" if (LIVE_MODE and pos.order_id) else "PAPER"
    log_event(f"[{mode} CLOSE/{reason}] {pos.question[:45]} | PnL=${pos.pnl:+.2f} entry={pos.entry_price:.3f} close={price:.3f}")

    if abs(pos.pnl) > 3 or reason == "STOP_LOSS":
        await send_alert(f"{'🟢' if pos.pnl >= 0 else '🔴'} {reason}\n{pos.question[:50]}\nPnL: ${pos.pnl:+.2f} | {pos.side} | {mode}")

    await append_trade_log({**asdict(pos), "event": "close", "reason": reason, "mode": mode, "ts": pos.close_time})

    # Save state immediately after every close — don't wait for next scan cycle
    await save_state()


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN SCAN CYCLE
# ──────────────────────────────────────────────────────────────────────────────
async def scan_cycle():
    state.scan_count += 1
    log_event(f"── Scan #{state.scan_count} [{'LIVE' if LIVE_MODE else 'PAPER'}] " + "─" * 30)

    markets = await fetch_markets(50)
    await update_positions(markets)

    open_cids = {p.condition_id for p in state.positions if p.status == "OPEN"}

    def question_allowed(q: str) -> bool:
        t = q.lower()
        sports_block = ["nba","nfl","fifa","epl","soccer","football","basketball","tennis","golf",
                        "cricket","ufc","boxing","baseball","hockey","mls","championship","league",
                        "cup","playoff","finals","tournament","match","score","vs ","beat "]
        if any(kw in t for kw in sports_block):
            return False
        return any(kw in t for kw in [
            "bitcoin","btc","ethereum","eth","crypto","solana","coinbase","binance","blockchain",
            "defi","nft","altcoin","stablecoin","price","above","below","reach","hit","close",
            "president","election","vote","prime minister","senate","congress","trump","policy",
            "war","ceasefire","iran","ukraine","china","russia","tariff","sanction","treaty",
            "nato","g7","g20","summit","nuclear","deal","resign",
            "fed","rate","inflation","gdp","stock","nasdaq","recession","oil","gold","interest",
            "cpi","fomc","s&p","earnings","bonds","yield","deficit","debt",
            "gpt","openai","claude","gemini","ai model","apple","google","microsoft","meta",
            "release","launch","announce","chipset","nvidia","nvda","regulation","ban","lawsuit",
        ])

    candidates = sorted(
    [m for m in markets
     if m.condition_id not in open_cids
     and m.volume > 10_000
     and m.yes_price >= 0.10
     and m.yes_price <= 0.90],     
    key=lambda m: m.volume, reverse=True,
)[:10]

    log_event(f"[SCAN] {len(candidates)} candidates from {len(markets)} markets")

    for market in candidates:
        state.signals_analyzed += 1
        signal = await analyse_market(market)
        if not signal:
            # signal=None means it was rejected inside analyse_market — already logged there
            continue
        allow, reason = risk_check(signal["edge"], signal["confidence"], market)
        if not allow:
            log_event(f"[SKIP] {market.question[:45]} — {reason} (edge={signal['edge']:.3f} conf={signal['confidence']:.2f})")
            continue
        await open_position(market, signal)
        await asyncio.sleep(2)

    prune_memory()
    await save_state()

    open_count = sum(1 for p in state.positions if p.status == "OPEN")
    log_event(f"Balance: ${state.balance:.2f} | PnL: ${state.total_pnl:+.2f} | Open: {open_count} | Trades: {state.trades_taken}")


# ──────────────────────────────────────────────────────────────────────────────
#  WEB DASHBOARD  (v3 — adds PnL integrity panel and ledger tab)
# ──────────────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PolyBot v3 Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#070b10;--s:#0d1520;--b:#172030;--t:#c8d8f0;--m:#4a5878;
      --a:#3b82f6;--g:#22c55e;--r:#ef4444;--y:#f59e0b;--p:#a855f7}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--t);min-height:100vh;padding:24px 20px}
.mono{font-family:'JetBrains Mono',monospace}
.hdr{display:flex;align-items:center;gap:12px;margin-bottom:4px}
h1{font-size:19px;font-weight:600;color:#fff}
.badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600}
.badge.live{background:#14532d;color:#86efac}.badge.paper{background:#1e3a5f;color:#93c5fd}
.badge.redis{background:#3b1f6e;color:#c4b5fd}.badge.json{background:#1c1917;color:#d6d3d1}
.dot{width:6px;height:6px;border-radius:50%;background:currentColor;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.sub{font-size:12px;color:var(--m);margin-bottom:20px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-bottom:20px}
.stat{background:var(--s);border:1px solid var(--b);border-radius:8px;padding:12px 14px}
.sl{font-size:10px;color:var(--m);text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px}
.sv{font-size:20px;font-weight:600;color:#fff;font-family:'JetBrains Mono',monospace}
.sv.g{color:var(--g)}.sv.r{color:var(--r)}.sv.b{color:var(--a)}.sv.p{color:var(--p)}
.integrity{background:var(--s);border:1px solid var(--b);border-radius:8px;padding:14px 16px;margin-bottom:16px}
.integrity.ok{border-color:#166534}.integrity.warn{border-color:#854d0e}.integrity.err{border-color:#991b1b}
.int-title{font-size:10px;font-weight:600;color:var(--m);text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px}
.int-row{display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px}
.int-label{color:var(--m)}.int-val{font-family:'JetBrains Mono',monospace;color:#fff}
.int-status{font-size:11px;font-weight:600;margin-top:8px;padding:4px 10px;border-radius:4px;display:inline-block}
.int-status.ok{background:#14532d;color:#86efac}.int-status.warn{background:#451a03;color:#fde68a}.int-status.na{background:#1c1917;color:#9ca3af}
.tabs{display:flex;gap:6px;margin-bottom:12px}
.tab{background:var(--s);border:1px solid var(--b);border-radius:5px;padding:5px 12px;font-size:11px;cursor:pointer;color:var(--m)}
.tab.active{background:var(--a);border-color:var(--a);color:#fff}
.sec{font-size:10px;font-weight:600;color:var(--m);text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px}
.tw{background:var(--s);border:1px solid var(--b);border-radius:8px;overflow:hidden;margin-bottom:16px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:9px 11px;color:var(--m);font-weight:500;border-bottom:1px solid var(--b);font-size:10px;text-transform:uppercase;letter-spacing:.06em}
td{padding:8px 11px;border-bottom:1px solid rgba(23,32,48,.5)}
tr:last-child td{border-bottom:none}
.pill{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600;font-family:'JetBrains Mono',monospace}
.pill.yes{background:#14532d;color:#86efac}.pill.no{background:#450a0a;color:#fca5a5}
.pill.ai{background:#1c1917;color:#d6d3d1}.pill.arb{background:#1e1b4b;color:#a5b4fc}.pill.mom{background:#172030;color:#7dd3fc}
.pp{color:var(--g)}.pn{color:var(--r)}
.live-dot{color:var(--g);font-size:11px}.paper-dot{color:var(--a);font-size:11px}
.log{background:var(--s);border:1px solid var(--b);border-radius:8px;padding:12px;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--m);max-height:280px;overflow-y:auto;line-height:1.7}
.le.err{color:#fca5a5}.le.ok{color:#86efac}.le.ai{color:#fde68a}
.tb{display:flex;gap:8px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
.btn{background:var(--a);color:#fff;border:none;border-radius:5px;padding:6px 14px;font-size:12px;cursor:pointer;font-family:inherit;font-weight:500}
.btn:hover{background:#1d4ed8}
.upd{font-size:11px;color:var(--m)}.empty{padding:14px;color:var(--m);font-size:12px;text-align:center}
.hidden{display:none}
</style>
</head>
<body>
<div class="hdr">
  <h1>PolyBot v3</h1>
  <span class="badge paper" id="mode-badge"><span class="dot"></span>PAPER</span>
  <span class="badge json" id="store-badge">JSON</span>
</div>
<p class="sub">3-signal AI trading · LLM + Momentum + Arb · Redis state persistence · auto-refresh 20s</p>
<div class="tb">
  <button class="btn" onclick="loadAll()">↻ Refresh</button>
  <span class="upd" id="upd">Loading…</span>
</div>

<div class="stats" id="stats"></div>

<!-- PnL Integrity Panel -->
<div class="integrity" id="integrity-box">
  <div class="int-title">PnL Integrity Check</div>
  <div class="int-row"><span class="int-label">Running total (state)</span><span class="int-val" id="int-running">—</span></div>
  <div class="int-row"><span class="int-label">Ledger sum (Redis)</span><span class="int-val" id="int-ledger">—</span></div>
  <div class="int-row"><span class="int-label">Drift</span><span class="int-val" id="int-drift">—</span></div>
  <span class="int-status na" id="int-status">Checking…</span>
</div>

<div class="tabs">
  <div class="tab active" onclick="showTab('positions')">Open Positions</div>
  <div class="tab" onclick="showTab('closed')">Closed Trades</div>
  <div class="tab" onclick="showTab('ledger')">PnL Ledger</div>
  <div class="tab" onclick="showTab('log')">Bot Log</div>
</div>

<div id="tab-positions">
  <div class="tw" id="pos-tbl"></div>
</div>
<div id="tab-closed" class="hidden">
  <div class="tw" id="cls-tbl"></div>
</div>
<div id="tab-ledger" class="hidden">
  <div class="tw" id="ledger-tbl"></div>
</div>
<div id="tab-log" class="hidden">
  <div class="log" id="log"></div>
</div>

<script>
const f=(n,d=2)=>(+n||0).toFixed(d);
const pc=n=>(+n>=0?'pp':'pn');
const ps=n=>(+n>=0?'+$':'−$')+f(Math.abs(n));
function srcPill(src){
  src=(src||'ai').toLowerCase();
  if(src.includes('arb'))return'<span class="pill arb">ARB</span>';
  if(src.includes('momentum'))return'<span class="pill mom">MOM</span>';
  return'<span class="pill ai">AI</span>';
}
function showTab(name){
  ['positions','closed','ledger','log'].forEach(t=>{
    document.getElementById('tab-'+t).classList.toggle('hidden',t!==name);
  });
  document.querySelectorAll('.tab').forEach((el,i)=>{
    el.classList.toggle('active',['positions','closed','ledger','log'][i]===name);
  });
}
async function loadAll(){
  await Promise.all([loadState(), loadLedger()]);
}
async function loadState(){
  try{
    const d=await(await fetch('/paper_trades.json?t='+Date.now())).json();
    document.getElementById('upd').textContent='Updated '+new Date(d.updated_at||Date.now()).toLocaleTimeString();
    const live=d.mode==='LIVE';
    const mb=document.getElementById('mode-badge');
    mb.innerHTML=`<span class="dot"></span>${d.mode||'PAPER'}`;
    mb.className='badge '+(live?'live':'paper');
    const sb=document.getElementById('store-badge');
    sb.textContent=d.storage||'JSON';
    sb.className='badge '+(d.storage==='Redis'?'redis':'json');

    const pnl=+d.total_pnl||0;
    document.getElementById('stats').innerHTML=`
      <div class="stat"><div class="sl">Balance</div><div class="sv mono">$${f(d.balance)}</div></div>
      <div class="stat"><div class="sl">Total PnL</div><div class="sv ${pc(pnl)} mono">${ps(pnl)}</div></div>
      <div class="stat"><div class="sl">Trades</div><div class="sv b">${d.trades_taken||0}</div></div>
      <div class="stat"><div class="sl">Open</div><div class="sv">${(d.open_positions||[]).length}</div></div>
      <div class="stat"><div class="sl">Scans</div><div class="sv">${d.scan_count||0}</div></div>
      <div class="stat"><div class="sl">Signals</div><div class="sv">${d.signals_analyzed||0}</div></div>`;

    // Update integrity running total
    document.getElementById('int-running').textContent='$'+f(pnl);

    const ops=d.open_positions||[];
    document.getElementById('pos-tbl').innerHTML=ops.length===0
      ?'<div class="empty">No open positions</div>'
      :`<table><thead><tr><th>Market</th><th>Side</th><th>Signals</th><th>Entry</th><th>Now</th><th>PnL</th><th>Conf</th><th>Mode</th></tr></thead>
       <tbody>${ops.map(p=>`<tr>
         <td style="max-width:190px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${p.question}">${p.question.slice(0,46)}${p.question.length>46?'…':''}</td>
         <td><span class="pill ${p.side.toLowerCase()}">${p.side}</span></td>
         <td>${srcPill(p.source)}</td>
         <td class="mono">${f(p.entry_price,3)}</td>
         <td class="mono">${f(p.current_price,3)}</td>
         <td class="mono ${pc(p.pnl)}">${ps(p.pnl)}</td>
         <td>${p.claude_confidence?f(p.claude_confidence*100,0)+'%':'—'}</td>
         <td>${p.order_id?'<span class="live-dot">● LIVE</span>':'<span class="paper-dot">◌ PAPER</span>'}</td>
       </tr>`).join('')}</tbody></table>`;

    const cl=(d.closed_trades||[]).slice(-20).reverse();
    document.getElementById('cls-tbl').innerHTML=cl.length===0
      ?'<div class="empty">No closed trades yet</div>'
      :`<table><thead><tr><th>Market</th><th>Side</th><th>Entry</th><th>Close</th><th>PnL</th><th>Reason</th><th>Time</th></tr></thead>
       <tbody>${cl.map(p=>`<tr>
         <td style="max-width:190px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${p.question}">${p.question.slice(0,46)}${p.question.length>46?'…':''}</td>
         <td><span class="pill ${p.side.toLowerCase()}">${p.side}</span></td>
         <td class="mono">${f(p.entry_price,3)}</td>
         <td class="mono">${f(p.close_price||0,3)}</td>
         <td class="mono ${pc(p.pnl)}">${ps(p.pnl)}</td>
         <td style="font-size:10px">${p.reason||'—'}</td>
         <td>${p.close_time?new Date(p.close_time).toLocaleTimeString():'-'}</td>
       </tr>`).join('')}</tbody></table>`;

    const ll=(d.log||[]).slice().reverse();
    document.getElementById('log').innerHTML=ll.map(l=>{
      let c='le';
      if(l.includes('ERROR')||l.includes('FAIL'))c+=' err';
      else if(l.includes('OPEN')||l.includes('PROFIT')||l.includes('✓'))c+=' ok';
      else if(l.includes('LLM')||l.includes('SIGNAL'))c+=' ai';
      return`<div class="${c}">${l}</div>`;
    }).join('')||'<div class="le">No log entries yet.</div>';
  }catch(e){
    document.getElementById('upd').textContent='Error: '+e.message;
  }
}
async function loadLedger(){
  try{
    const d=await(await fetch('/ledger?limit=50&t='+Date.now())).json();
    const ledger=d.entries||[];
    const ledgerTotal=d.ledger_total;
    const runningTotal=+(document.getElementById('int-running').textContent.replace(/[^0-9.-]/g,''))||null;

    document.getElementById('int-ledger').textContent=
      ledgerTotal!=null?'$'+f(ledgerTotal):'Not available (no Redis)';

    const box=document.getElementById('integrity-box');
    const statusEl=document.getElementById('int-status');
    if(ledgerTotal==null){
      document.getElementById('int-drift').textContent='N/A';
      statusEl.textContent='Redis not connected';
      statusEl.className='int-status na';
      box.className='integrity';
    } else {
      const drift=Math.abs((runningTotal||0)-ledgerTotal);
      document.getElementById('int-drift').textContent='$'+f(drift);
      if(drift<0.01){
        statusEl.textContent='✓ PnL verified — no drift';
        statusEl.className='int-status ok';
        box.className='integrity ok';
      } else {
        statusEl.textContent=`⚠ Drift detected: $${f(drift)} — check trade_history.jsonl`;
        statusEl.className='int-status warn';
        box.className='integrity warn';
      }
    }

    document.getElementById('ledger-tbl').innerHTML=ledger.length===0
      ?'<div class="empty">No ledger entries yet</div>'
      :`<table><thead><tr><th>Time</th><th>Market</th><th>Side</th><th>Entry</th><th>Close</th><th>PnL</th><th>Reason</th><th>Mode</th></tr></thead>
       <tbody>${ledger.map(e=>`<tr>
         <td style="font-size:10px">${e.ts?new Date(e.ts).toLocaleTimeString():'-'}</td>
         <td style="max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${e.question||''}">${(e.question||'').slice(0,36)}${(e.question||'').length>36?'…':''}</td>
         <td><span class="pill ${(e.side||'').toLowerCase()}">${e.side||'—'}</span></td>
         <td class="mono">${f(e.entry_price||0,3)}</td>
         <td class="mono">${f(e.close_price||0,3)}</td>
         <td class="mono ${pc(e.pnl)}">${ps(e.pnl)}</td>
         <td style="font-size:10px">${e.reason||'—'}</td>
         <td style="font-size:10px">${e.mode||'—'}</td>
       </tr>`).join('')}</tbody></table>`;
  }catch(e){
    console.warn('Ledger load failed:', e.message);
  }
}
loadAll();
setInterval(loadAll,20000);
</script>
</body>
</html>"""


async def _handle_root(request):
    return aio_web.Response(text=DASHBOARD_HTML, content_type="text/html")


async def _handle_trades(request):
    """Serves current state as JSON — used by the dashboard."""
    r = await get_redis()
    storage = "Redis" if (r is not None and _redis_available) else "JSON"

    base = {
        "balance":          state.balance,
        "total_pnl":        state.total_pnl,
        "scan_count":       state.scan_count,
        "signals_analyzed": state.signals_analyzed,
        "trades_taken":     state.trades_taken,
        "start_time":       state.start_time,
        "mode":             "LIVE" if LIVE_MODE else "PAPER",
        "storage":          storage,
        "open_positions":   [asdict(p) for p in state.positions if p.status == "OPEN"],
        "closed_trades":    state.closed_trades[-100:],
        "log":              state.log[-200:],
        "updated_at":       datetime.now(timezone.utc).isoformat(),
    }
    return aio_web.json_response(base)


async def _handle_ledger(request):
    """
    Returns the immutable PnL ledger from Redis.
    Used by the dashboard integrity check.
    """
    limit = int(request.rel_url.query.get("limit", 100))
    ledger_total = await redis_get_ledger_total()
    entries      = await redis_get_ledger_entries(limit)
    return aio_web.json_response({
        "ledger_total": ledger_total,
        "entry_count":  len(entries),
        "entries":      entries,
    })


async def _handle_health(request):
    r = await get_redis()
    redis_ok = r is not None and _redis_available
    open_count = sum(1 for p in state.positions if p.status == "OPEN")
    return aio_web.json_response({
        "status":         "ok",
        "mode":           "LIVE" if LIVE_MODE else "PAPER",
        "storage":        "redis" if redis_ok else "json",
        "scans":          state.scan_count,
        "open_positions": open_count,
        "balance":        round(state.balance, 2),
        "total_pnl":      round(state.total_pnl, 4),
        "trades":         state.trades_taken,
    })


async def start_web():
    app = aio_web.Application()
    app.router.add_get("/",                  _handle_root)
    app.router.add_get("/paper_trades.json", _handle_trades)
    app.router.add_get("/ledger",            _handle_ledger)
    app.router.add_get("/health",            _handle_health)
    runner = aio_web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv("PORT", 8080))
    await aio_web.TCPSite(runner, "0.0.0.0", port).start()
    log_event(f"Dashboard → http://0.0.0.0:{port}")


async def self_ping():
    await asyncio.sleep(90)
    port = int(os.getenv("PORT", 8080))
    while True:
        try:
            http = await get_http()
            await http.get(f"http://localhost:{port}/health")
        except Exception:
            pass
        await asyncio.sleep(600)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN BOT LOOP
# ──────────────────────────────────────────────────────────────────────────────
async def bot_loop():
    await start_web()
    asyncio.create_task(self_ping())
    asyncio.create_task(ws_price_feed())

    # ── Restore state BEFORE printing startup stats ──
    load_saved_state()

    mode_str = "🔴 LIVE TRADING — REAL MONEY" if LIVE_MODE else "📄 PAPER TRADING (safe)"
    r = await get_redis()
    store_str = f"Redis ({REDIS_URL[:25]}…)" if (r is not None and _redis_available) else "JSON file (no Redis)"

    log_event(f"Polymarket AI Bot v3 started — {mode_str}")
    log_event(f"State storage: {store_str}")
    log_event(f"LLM: Replicate → {REPLICATE_MODEL}")
    log_event(f"Signals: LLM (0.50) + Momentum (0.35) + Spread-Arb (0.15)")
    log_event(f"Balance: ${state.balance:.2f} | PnL: ${state.total_pnl:+.2f} | Trades: {state.trades_taken}")
    log_event("─" * 60)

    if LIVE_MODE:
        live_bal = await clob_get_balance()
        if live_bal > 0:
            state.balance = live_bal
            log_event(f"[LIVE] Chain balance synced: ${live_bal:.2f} USDC")
        else:
            log_event("[LIVE WARNING] Could not fetch on-chain balance — using saved value")
        await send_alert(
            f"🤖 PolyBot v3 LIVE started\nBalance: ${state.balance:.2f} USDC\n"
            f"Total PnL to date: ${state.total_pnl:+.2f}\n"
            f"Storage: {store_str}"
        )

    while True:
        try:
            await scan_cycle()
        except Exception as e:
            log_event(f"[LOOP ERROR] {e}")
            await send_alert(f"⚠️ Bot loop error: {e}")
        await asyncio.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    asyncio.run(bot_loop())