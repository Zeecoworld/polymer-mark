"""
Polymarket AI Trading Bot — v4 (Smart Entry Edition)
=====================================================

CORE STRATEGY — "Buy Low, Sell High on Prediction Markets":
  - Scans 200 markets every cycle and RANKS them by opportunity score
  - Only enters positions priced between 0.10–0.45 (cheap YES) or cheap NO equivalent
  - Waits for price to drift toward 0.70–0.90 range before taking profit
  - Much wider take-profit window, tighter stop to protect capital
  - Never bets on sports, memes, jokes, or joke markets
  - Blocks contradicting positions on same person/topic (no self-hedging)
  - LLM confidence variance forced — no more 0.55 collapse
  - Dynamic stop-loss based on entry price (protects expensive entries)
  - Semantic deduplication across related markets
  - MAX_OPEN_POSITIONS = 5 (quality over quantity)
  - Position sizing scales with conviction — high-edge trades get more capital

BUY LOW / SELL HIGH MECHANIC:
  - We only enter when market price is "cheap" (0.10–0.45)
  - Take profit triggers when price reaches entry + 0.25 minimum (up to 0.90)
  - This gives a natural 2.5:1 reward-to-risk ratio
  - Markets near resolution with strong LLM backing get priority

REQUIRED ENV VARS (paper mode):
    REPLICATE_API_KEY      — get at replicate.com/account/api-tokens

REDIS (optional but strongly recommended):
    REDIS_URL              — e.g. redis://localhost:6379

ADDITIONAL ENV VARS (live mode, LIVE_MODE=true):
    POLYMARKET_PRIVATE_KEY
    POLYMARKET_API_KEY
    POLYMARKET_API_SECRET
    POLYMARKET_API_PASSPHRASE
    FUNDER_ADDRESS
    SIGNATURE_TYPE         — 0=EOA/MetaMask, 1=Email/Magic (default 1)

OPTIONAL:
    TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
    SLACK_WEBHOOK_URL
    PAPER_BALANCE          — starting paper balance (default 1000)
    MAX_OPEN_POSITIONS     — default 5 (DO NOT go above 8)
    SCAN_INTERVAL          — seconds between scans (default 120)
    MIN_EDGE_PCT           — minimum edge to trade (default 0.04)
    MAX_POSITION_PCT       — max balance per trade (default 0.08)
    PORT                   — dashboard port (default 8080)

REDIS KEYS USED:
    polybot:state          — main state snapshot
    polybot:positions      — open/closed Position objects
    polybot:closed_trades  — list of closed trade dicts (capped at 500)
    polybot:pnl_ledger     — append-only PnL event log (never deleted)
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

# ── RISK CONFIG — tuned for Buy-Low/Sell-High strategy ──
PAPER_BALANCE_USDC     = float(os.getenv("PAPER_BALANCE",      "1000.0"))
SCAN_INTERVAL          = int(os.getenv("SCAN_INTERVAL",        "120"))
MAX_POSITION_PCT       = float(os.getenv("MAX_POSITION_PCT",   "0.08"))   # up to 8% per trade
MIN_EDGE_PCT           = float(os.getenv("MIN_EDGE_PCT",        "0.04"))  # 4% minimum edge
MAX_OPEN_POSITIONS     = int(os.getenv("MAX_OPEN_POSITIONS",    "5"))     # quality > quantity
COOLDOWN_SECONDS       = int(os.getenv("COOLDOWN_SECONDS",     "3600"))  # 1hr base cooldown

# ── BUY-LOW / SELL-HIGH PRICE BANDS ──
# We only enter "cheap" positions — price must be in this range
ENTRY_MIN_PRICE        = 0.12   # don't buy below 12¢ (too speculative)
ENTRY_MAX_PRICE        = 0.45   # don't buy above 45¢ (not cheap enough)
# Take profit when price reaches this level
TAKE_PROFIT_TARGET     = 0.72   # sell at 72¢+ (was entry+0.30 — now absolute target)
TAKE_PROFIT_MIN_GAIN   = 0.22   # minimum gain before selling (even if below 72¢)
# Stop loss
STOP_LOSS_PCT          = 0.40   # exit if price drops 40% of entry (e.g. enter 0.30 → stop 0.18)
STOP_LOSS_FLOOR        = 0.07   # never let position drop below 7¢

# Alerts
TELEGRAM_TOKEN         = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID       = os.getenv("TELEGRAM_CHAT_ID", "")
SLACK_WEBHOOK          = os.getenv("SLACK_WEBHOOK_URL", "")

# Persistence
REDIS_URL              = os.getenv("REDIS_URL", "")
LOG_FILE               = os.getenv("LOG_FILE", "paper_trades.json")
TRADE_LOG_FILE         = "trade_history.jsonl"

# Redis keys
RK_STATE          = "polybot:state"
RK_POSITIONS      = "polybot:positions"
RK_CLOSED_TRADES  = "polybot:closed_trades"
RK_PNL_LEDGER     = "polybot:pnl_ledger"
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
#  MARKET FILTERS — the heart of "don't enter bad trades"
# ──────────────────────────────────────────────────────────────────────────────

# ── Sports block — expanded to catch all variants ──
SPORTS_BLOCK_KEYWORDS = [
    "nba","nfl","fifa","epl","soccer","football","basketball","tennis","golf",
    "cricket","ufc","boxing","baseball","hockey","mls","championship",
    "cup","playoff","finals","tournament","match","score",
    # Club names
    "arsenal","chelsea","liverpool","manchester","tottenham","barcelona",
    "real madrid","juventus","bayern","psg","inter milan","ac milan",
    "avalanche","thunder","celtics","lakers","knicks","nuggets","suns",
    "warriors","heat","bucks","76ers","raptors","clippers","spurs",
    "cowboys","patriots","chiefs","eagles","rams","broncos","49ers",
    # Explicit competition phrases
    "premier league","champions league","serie a","bundesliga","ligue 1",
    "stanley cup","world cup","super bowl","march madness","ncaa",
    "win the 2026 n","win the nhl","win the nba","win the nfl","win the mlb",
    "win the premier","win the champions","win the stanley",
    "win the super bowl","win the world series","win the masters",
]

# ── Meme/joke/religious/sci-fi markets — LLM has no real edge ──
MEME_BLOCK_KEYWORDS = [
    "jesus","christ","god","allah","buddha","holy","divine","rapture",
    "alien","ufo","extraterrestrial","area 51","bigfoot","loch ness",
    "flat earth","zombie","apocalypse","end of the world","second coming",
    "gta vi","gta 6","gta6","half-life 3","elder scrolls 6",
    "time travel","parallel universe","simulation","matrix",
    "elon musk on mars","first person on mars","mars colony",
]

# ── Trusted high-edge categories for buy-low strategy ──
PREFERRED_CATEGORIES = [
    # Crypto / finance — fast moving, good price swings
    "bitcoin","btc","ethereum","eth","crypto","solana","sol","coinbase",
    "binance","blockchain","defi","altcoin","stablecoin",
    "price above","price below","reach $","hit $","close above","close below",
    "fed","rate cut","rate hike","inflation","gdp","recession","interest rate",
    "nasdaq","s&p","oil price","gold price","fomc","cpi","earnings",
    # Geopolitics — binary outcomes, clear resolution
    "ceasefire","peace deal","sanctions","treaty","war","invasion","nuclear",
    "nato","g7","g20","summit","resign","impeach","arrest","indicted",
    "ukraine","russia","iran","china","taiwan","north korea","israel","hamas",
    "tariff","trade deal","trade war",
    # US Politics — high volume, clear resolution dates
    "president","election","vote","senate","congress","house","supreme court",
    "trump","biden","harris","newsom","desantis","vance","rubio",
    "win the 2028","win the 2026","win the election","win the primary",
    # Tech — binary product/regulatory events
    "gpt","openai","claude","gemini","ai model","apple","google","microsoft",
    "meta","release","launch","announce","ban","lawsuit","regulation",
    "nvidia","nvda","chipset","ipo","acquisition","merger",
    # Climate / science — clear resolution
    "hurricane","earthquake","flood","temperature record",
    "greenland","arctic","climate","co2",
]


def question_allowed(q: str) -> bool:
    """
    Three-gate filter:
    1. Block sports markets (LLM has no edge vs sports bettors)
    2. Block meme/joke/religious markets (no reliable resolution signal)
    3. Require at least one preferred keyword (stick to high-edge categories)
    """
    t = q.lower()

    # Gate 1 — sports block
    if any(kw in t for kw in SPORTS_BLOCK_KEYWORDS):
        return False

    # Gate 2 — meme/joke block
    if any(kw in t for kw in MEME_BLOCK_KEYWORDS):
        return False

    # Gate 3 — must match at least one preferred category
    if any(kw in t for kw in PREFERRED_CATEGORIES):
        return True

    return False


def _question_too_similar(new_q: str, open_positions: list) -> bool:
    """
    Semantic deduplication — prevents opening contradicting positions
    on the same person, country, or topic.
    Examples caught:
      - YES "JD Vance wins 2028 Republican primary"
      - NO  "JD Vance wins 2028 US election"
    Both are about Vance's electoral prospects — we should only hold one.
    """
    new_lower = new_q.lower()
    open_questions = [p.question.lower() for p in open_positions if p.status == "OPEN"]

    # Key entities — if new question AND an existing position both contain
    # the same entity, block the new one
    entity_groups = [
        # People
        ["vance","jd vance","j.d. vance"],
        ["trump","donald trump"],
        ["newsom","gavin newsom"],
        ["rubio","marco rubio"],
        ["harris","kamala harris"],
        ["biden","joe biden"],
        ["desantis","ron desantis"],
        ["modi","narendra modi"],
        ["putin","vladimir putin"],
        ["zelensky","volodymyr zelensky"],
        ["netanyahu","benjamin netanyahu"],
        ["xi jinping","xi jinping"],
        # Countries / regions
        ["iran","iranian"],
        ["ukraine","ukrainian"],
        ["russia","russian"],
        ["china","chinese","taiwan"],
        ["israel","israeli","hamas","gaza"],
        ["greenland"],
        ["north korea","kim jong"],
        # Topics
        ["bitcoin","btc"],
        ["ethereum","eth"],
        ["fed rate","federal reserve","fomc"],
        ["strait of hormuz","hormuz"],
        ["world cup 2026","2026 world cup","fifa 2026"],
    ]

    for group in entity_groups:
        new_has = any(ent in new_lower for ent in group)
        if not new_has:
            continue
        for oq in open_questions:
            oq_has = any(ent in oq for ent in group)
            if oq_has:
                return True  # already have a position on this entity

    return False


# ──────────────────────────────────────────────────────────────────────────────
#  REDIS CLIENT
# ──────────────────────────────────────────────────────────────────────────────
_redis = None
_redis_available = False


async def get_redis():
    global _redis, _redis_available
    if _redis is not None:
        return _redis if _redis_available else None
    if not REDIS_URL:
        return None
    try:
        import redis.asyncio as aioredis
        _redis = aioredis.from_url(
            REDIS_URL, encoding="utf-8", decode_responses=True,
            socket_connect_timeout=3, socket_timeout=3, retry_on_timeout=True,
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


async def redis_save_state(state_obj) -> bool:
    r = await get_redis()
    if r is None:
        return False
    try:
        pipe = r.pipeline()
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
        pipe.set(RK_POSITIONS,     json.dumps([asdict(p) for p in state_obj.positions]))
        pipe.set(RK_CLOSED_TRADES, json.dumps(state_obj.closed_trades[-500:]))
        pipe.set(RK_LOG,           json.dumps(state_obj.log[-200:]))
        await pipe.execute()
        return True
    except Exception as e:
        print(f"[REDIS] save_state failed: {e}")
        return False


async def redis_load_state() -> Optional[dict]:
    r = await get_redis()
    if r is None:
        return None
    try:
        pipe = r.pipeline()
        pipe.get(RK_STATE); pipe.get(RK_POSITIONS)
        pipe.get(RK_CLOSED_TRADES); pipe.get(RK_LOG)
        results = await pipe.execute()
        if not results[0]:
            return None
        return {
            **json.loads(results[0]),
            "positions":     json.loads(results[1]) if results[1] else [],
            "closed_trades": json.loads(results[2]) if results[2] else [],
            "log":           json.loads(results[3]) if results[3] else [],
        }
    except Exception as e:
        print(f"[REDIS] load_state failed: {e}")
        return None


async def redis_append_pnl_ledger(entry: dict):
    r = await get_redis()
    if r is None:
        return
    try:
        await r.rpush(RK_PNL_LEDGER, json.dumps({
            **entry, "ts": datetime.now(timezone.utc).isoformat(),
        }))
    except Exception as e:
        print(f"[REDIS] append_pnl_ledger failed: {e}")


async def redis_get_ledger_total() -> Optional[float]:
    r = await get_redis()
    if r is None:
        return None
    try:
        entries = await r.lrange(RK_PNL_LEDGER, 0, -1)
        return round(sum(float(json.loads(e).get("pnl", 0)) for e in entries), 4)
    except Exception as e:
        print(f"[REDIS] get_ledger_total failed: {e}")
        return None


async def redis_get_ledger_entries(limit: int = 100) -> list:
    r = await get_redis()
    if r is None:
        return []
    try:
        entries = await r.lrange(RK_PNL_LEDGER, -limit, -1)
        return [json.loads(e) for e in reversed(entries)]
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
#  CLOB CLIENT  (live mode only)
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
        limit_price  = round(min(price + 0.01, 0.99), 3)
        size_shares  = round(size_usdc / limit_price, 4)
        def _place():
            client = get_clob_client()
            order  = client.create_order(OrderArgs(token_id=token_id, price=limit_price, size=size_shares, side=BUY))
            return client.post_order(order, OrderType.GTC)
        result = await asyncio.to_thread(_place)
        log_event(f"[CLOB ORDER] token={token_id[:14]}… price={limit_price} shares={size_shares} status={result.get('status','?')}")
        return result
    except Exception as e:
        log_event(f"[CLOB ERROR] Limit order failed: {e}")
        return None


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
_ws_prices:     dict[str, float] = {}
_ws_subscribed: set[str]         = set()


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
    "politics": [
        "https://news.google.com/rss/search?q=US+politics+election+2026&hl=en",
        "https://feeds.reuters.com/reuters/politicsNews",
    ],
    "finance": [
        "https://news.google.com/rss/search?q=federal+reserve+interest+rates&hl=en",
        "https://feeds.reuters.com/reuters/businessNews",
    ],
}

_news_cache:      dict = {}
_news_cache_time: dict = {}
NEWS_CACHE_TTL        = 300
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
        r    = await http.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return []
        root      = ET.fromstring(r.text)
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
    q   = question.lower()
    if any(w in q for w in ["bitcoin","btc","eth","crypto","solana","defi","nft","binance","coinbase"]):
        feed_keys = ["crypto", "general"]
    elif any(w in q for w in ["gpt","ai","openai","apple","google","microsoft","meta","nvidia","chipset"]):
        feed_keys = ["tech", "general"]
    elif any(w in q for w in ["president","election","senate","congress","vote","trump","harris","newsom","vance","rubio"]):
        feed_keys = ["politics", "general"]
    elif any(w in q for w in ["fed","rate","inflation","gdp","recession","nasdaq","s&p","earnings","fomc","cpi"]):
        feed_keys = ["finance", "general"]
    else:
        feed_keys = ["general"]

    all_headlines: list[str] = []
    for key in feed_keys[:2]:
        for url in RSS_FEEDS.get(key, [])[:2]:
            if url in _news_cache and now - _news_cache_time.get(url, 0) < NEWS_CACHE_TTL:
                all_headlines.extend(_news_cache[url])
            else:
                h                    = await fetch_rss(url)
                _news_cache[url]     = h
                _news_cache_time[url]= now
                all_headlines.extend(h)

    if not all_headlines:
        return "No recent news found."
    keywords = [w for w in question.replace("?","").lower().split() if len(w) > 3]
    def relevance(h: str) -> int:
        return sum(1 for kw in keywords if kw in h.lower())
    scored = sorted(set(all_headlines), key=relevance, reverse=True)
    top    = [h for h in scored if relevance(h) > 0][:5] or scored[:3]
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
        await http.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": f"🤖 PolyBot v4\n{msg}", "parse_mode": "Markdown"},
        )
    except Exception as e:
        log_event(f"[ALERT] Telegram failed: {e}")

async def _slack_alert(msg: str):
    try:
        http = await get_http()
        await http.post(SLACK_WEBHOOK, json={"text": f"🤖 PolyBot v4: {msg}"})
    except Exception as e:
        log_event(f"[ALERT] Slack failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
#  DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────
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
    opportunity_score: float = 0.0   # NEW — ranks quality of entry


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
_recently_closed: dict[str, float] = {}


# ──────────────────────────────────────────────────────────────────────────────
#  STATE PERSISTENCE
# ──────────────────────────────────────────────────────────────────────────────
def load_saved_state():
    loop = asyncio.get_event_loop()
    try:
        saved = loop.run_until_complete(redis_load_state())
    except Exception:
        saved = None

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

    state.balance          = float(saved.get("balance",          PAPER_BALANCE_USDC))
    state.total_pnl        = float(saved.get("total_pnl",        0.0))
    state.scan_count       = int(saved.get("scan_count",         0))
    state.signals_analyzed = int(saved.get("signals_analyzed",   0))
    state.trades_taken     = int(saved.get("trades_taken",       0))
    state.start_time       = saved.get("start_time",             state.start_time)
    state.closed_trades    = saved.get("closed_trades",          [])
    state.log              = saved.get("log",                    [])

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
        f"balance=${state.balance:.2f} total_pnl=${state.total_pnl:+.2f} "
        f"trades={state.trades_taken} "
        f"open={len([p for p in state.positions if p.status == 'OPEN'])}"
    )


def _write_json_fallback():
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
    redis_ok = await redis_save_state(state)
    await asyncio.to_thread(_write_json_fallback)
    return redis_ok


async def append_trade_log(trade: dict):
    async with aiofiles.open(TRADE_LOG_FILE, "a", encoding="utf-8") as f:
        await f.write(json.dumps(trade, ensure_ascii=False) + "\n")
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
async def fetch_markets(limit: int = 200) -> list[Market]:
    try:
        http = await get_http()
        r    = await http.get(f"{GAMMA_API}/markets", params={
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
    """
    Mock markets designed to test the buy-low strategy.
    All YES prices are in the 0.12–0.45 buy zone.
    """
    base = [
        ("Will BTC close above $100k before June 2026?",       0.28, 0.72, 2_100_000, "crypto"),
        ("Will ETH hit $3k before May 2026?",                  0.32, 0.68,   890_000, "crypto"),
        ("Will Fed cut rates in May 2026?",                    0.22, 0.78, 1_200_000, "finance"),
        ("Will Russia x Ukraine ceasefire be signed by 2026?", 0.38, 0.62, 1_800_000, "politics"),
        ("Will NVDA stock be above $900 by June 2026?",        0.41, 0.59,   560_000, "finance"),
        ("Will US-Iran peace deal happen before 2027?",        0.14, 0.86,   320_000, "politics"),
        ("Will US enter recession by Q3 2026?",                0.35, 0.65,   780_000, "finance"),
        ("Will OpenAI release GPT-5 before June 2026?",       0.44, 0.56,   290_000, "tech"),
        ("Will Trump impose 50%+ tariffs on China in 2026?",   0.39, 0.61,   450_000, "politics"),
        ("Will Greenland hold independence referendum 2026?",  0.17, 0.83,   210_000, "politics"),
    ]
    now = datetime.now(timezone.utc)
    return [
        Market(
            id=f"mock_{i}", condition_id=f"mock_cid_{i}",
            question=q,
            yes_price=round(yp + random.uniform(-0.02, 0.02), 3),
            no_price=round(np_ + random.uniform(-0.02, 0.02), 3),
            volume=vol,
            end_date=(now + timedelta(days=random.randint(3, 60))).isoformat(),
            category=cat,
        )
        for i, (q, yp, np_, vol, cat) in enumerate(base)
    ]


async def fetch_single_market_price(market_id: str, side: str) -> Optional[float]:
    try:
        http = await get_http()
        r    = await http.get(f"{GAMMA_API}/markets/{market_id}")
        r.raise_for_status()
        m = r.json()
        op  = m.get("outcomePrices")
        out = m.get("outcomes", '["Yes","No"]')
        if op:
            prices   = json.loads(op)  if isinstance(op, str) else op
            outcomes = json.loads(out) if isinstance(out, str) else out
            if side == "YES":
                yi = next((i for i, o in enumerate(outcomes) if o.lower() in ("yes","true")), 0)
                return float(prices[yi])
            else:
                ni = next((i for i, o in enumerate(outcomes) if o.lower() in ("no","false")), 1)
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
    x_mean = (n - 1) / 2
    y_mean = sum(hist) / n
    num = sum((i - x_mean) * (hist[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return None
    slope = num / den
    if abs(slope) < 0.005:   # tighter: only act on real momentum
        return None
    return round(max(0.02, min(0.98, current_price + slope * 3)), 4)


# ──────────────────────────────────────────────────────────────────────────────
#  LLM ANALYSIS  — v4 fixes confidence collapse + buy-low framing
# ──────────────────────────────────────────────────────────────────────────────
async def call_replicate_llama(prompt: str) -> str:
    system    = "You are a quantitative prediction market analyst. Always respond with valid JSON only. No markdown fences, no extra text."
    formatted = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    headers = {
        "Authorization": f"Bearer {REPLICATE_API_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "wait",
    }
    payload = {
        "input": {
            "prompt":      formatted,
            "max_tokens":  350,
            "temperature": 0.25,
            "stop":        ["<|eot_id|>", "<|end_of_text|>"],
        }
    }
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


async def analyse_market_llm(market: Market) -> Optional[dict]:
    """
    LLM analysis with v4 improvements:
    1. Confidence variance forced — no more 0.55 collapse
    2. Framed as "buy low" opportunity question
    3. Resolution timeline factored into confidence
    4. Explicit confidence rubric in prompt
    """
    news      = await get_news_for_market(market.question, market.category)
    days_left = "unknown"
    days_int  = 999
    try:
        end       = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
        days_int  = max(0, (end - datetime.now(timezone.utc)).days)
        days_left = str(days_int)
    except Exception:
        pass

    # Compute implied probability gap — this is what we're looking for
    implied_yes = market.yes_price * 100

    prompt = (
        f"You are a prediction market trader hunting for mispriced markets.\n\n"
        f"Question: {market.question}\n"
        f"Category: {market.category}\n"
        f"Days until resolution: {days_left}\n"
        f"Current YES price: ${market.yes_price:.3f} ({implied_yes:.1f}% implied probability)\n"
        f"Current NO price:  ${market.no_price:.3f}\n"
        f"24h Volume: ${market.volume:,.0f}\n"
        f"Description: {market.description or 'N/A'}\n"
        f"Recent headlines: {news}\n\n"
        f"YOUR TASK:\n"
        f"1. Estimate the TRUE probability that YES resolves (0.01 to 0.99)\n"
        f"2. Rate your confidence based on how much relevant info you have:\n"
        f"   - 0.75-0.90: Strong recent news directly about this topic\n"
        f"   - 0.60-0.74: Some relevant context, moderate certainty\n"
        f"   - 0.45-0.59: Limited relevant info, educated guess\n"
        f"   - 0.30-0.44: Very little info, speculative\n"
        f"   NEVER default to 0.55 — use the actual evidence quality to set confidence.\n"
        f"3. Calculate the edge: true_prob_yes minus market YES price\n"
        f"   Positive edge = YES is cheap (buy YES). Negative edge = NO is cheap (buy NO).\n\n"
        f"IMPORTANT: Be willing to disagree with the market. "
        f"If the market says 28% but you think it's 45% based on news, say so.\n\n"
        f"Respond ONLY with this exact JSON:\n"
        '{"true_prob_yes": 0.45, "confidence": 0.68, "reasoning": "one specific sentence with evidence"}\n'
    )

    text = ""
    try:
        text = await call_replicate_llama(prompt)
        log_event(f"[LLM RAW] {text[:120]}")

        # Strip markdown fences if present
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        s, e_idx = text.find("{"), text.rfind("}") + 1
        if s == -1 or e_idx <= s:
            return None
        data = json.loads(text[s:e_idx])

        true_prob = float(data.get("true_prob_yes", market.yes_price))
        conf      = float(data.get("confidence",    0.50))
        reasoning = data.get("reasoning", "")

        # ── Sanity clamp — reject extreme outputs ──
        if true_prob < 0.03 or true_prob > 0.97:
            # Extreme prediction with no strong news → reduce confidence
            if conf < 0.75:
                log_event(f"[LLM] Extreme prob {true_prob:.2f} rejected — no strong news backing")
                return None
            # If high confidence at extremes, still allow but cap
            true_prob = max(0.04, min(0.96, true_prob))

        true_prob = max(0.01, min(0.99, true_prob))
        conf      = max(0.30, min(0.95, conf))

        # ── Penalise far-future markets (>180 days) — too uncertain ──
        if days_int > 180:
            conf = max(0.30, conf - 0.12)
            log_event(f"[LLM] Confidence reduced for far-future market ({days_int}d): {conf:.2f}")

        # ── Boost confidence when headlines are highly relevant ──
        if news != "No relevant news found." and news != "No recent news found.":
            words_in_question = set(market.question.lower().split())
            headline_words    = set(news.lower().split())
            overlap           = len(words_in_question & headline_words)
            if overlap > 3:
                conf = min(0.92, conf + 0.05)

        return {
            "true_prob":  true_prob,
            "confidence": round(conf, 3),
            "reasoning":  reasoning,
            "source":     "llm",
        }
    except Exception as e:
        log_event(f"[LLM ERROR] {market.question[:45]}: {e}")
        log_event(f"[LLM RAW]  {text[:120] if text else 'no response'}")
        return None


def score_opportunity(market: Market, signal: dict) -> float:
    """
    Score how good a buy-low opportunity this is.
    Higher = better. Used to rank candidates before picking the best N.

    Factors:
    - Edge magnitude (how mispriced is it?)
    - Confidence (how sure are we?)
    - Entry price (cheaper = more upside room to profit)
    - Days to resolution (shorter = faster profit realisation)
    - Volume (more liquid = easier to exit)
    """
    edge       = abs(signal.get("edge",       0))
    confidence = signal.get("confidence",     0)
    entry      = signal.get("entry_price",    0.5)
    volume     = market.volume

    # Upside room: cheap entry has more room to grow toward 0.70+
    # e.g. entry=0.15 → room=0.57, entry=0.40 → room=0.32
    upside_room = max(0, TAKE_PROFIT_TARGET - entry)

    # Days bonus — resolve sooner = faster profit
    days_bonus = 0
    try:
        end  = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
        days = max(1, (end - datetime.now(timezone.utc)).days)
        # Sweet spot: 7–60 days. Too short (<7) = risky, too long (>180) = capital locked
        if 7 <= days <= 60:
            days_bonus = 0.15
        elif days <= 120:
            days_bonus = 0.05
        else:
            days_bonus = -0.10
    except Exception:
        pass

    # Volume score — normalise to 0–0.1
    vol_score = min(0.10, (volume / 5_000_000) * 0.10)

    score = (
        edge       * 0.35 +
        confidence * 0.30 +
        upside_room* 0.20 +
        vol_score  * 0.10 +
        days_bonus * 0.05
    )

    return round(score, 4)


def detect_spread_arb(market: Market) -> Optional[dict]:
    """Spread Arb: when YES+NO < 0.97, guaranteed gap exists regardless of outcome."""
    spread = market.yes_price + market.no_price
    if spread < 0.97:
        gap      = 1.0 - spread
        # Buy the cheaper side
        side     = "YES" if market.yes_price <= market.no_price else "NO"
        token_id = market.yes_token_id if side == "YES" else market.no_token_id
        return {"type": "spread_arb", "gap": gap, "side": side, "token_id": token_id}
    return None


def detect_overpriced_spread(market: Market) -> Optional[dict]:
    """Logical Arb: when YES+NO > 1.05, buy the cheaper side."""
    spread = market.yes_price + market.no_price
    if spread > 1.05:
        side     = "NO" if market.no_price < market.yes_price else "YES"
        edge     = (spread - 1.0) * 0.7
        token_id = market.no_token_id if side == "NO" else market.yes_token_id
        return {"type": "overpriced_arb", "gap": spread - 1.0, "edge": round(edge, 4), "side": side, "token_id": token_id}
    return None


async def analyse_market(market: Market) -> Optional[dict]:
    """
    Multi-signal analysis. Returns the best signal dict if actionable, else None.

    Signal sources:
      1. LLM AI (weight 0.40) — primary alpha source
      2. Spread Arb (weight 0.30) — near risk-free
      3. Logical Arb (weight 0.15) — overpriced spread
      4. Momentum (weight 0.15) — trend following
    """
    record_price(market.condition_id, market.yes_price)
    signals = []

    # ── 1. LLM AI ──
    llm_result = await analyse_market_llm(market)
    if llm_result and llm_result["confidence"] >= 0.35:
        edge_yes = llm_result["true_prob"] - market.yes_price
        edge_no  = (1 - llm_result["true_prob"]) - market.no_price
        side     = "YES" if edge_yes >= edge_no else "NO"
        edge     = edge_yes if side == "YES" else edge_no
        if abs(edge) >= MIN_EDGE_PCT:
            signals.append({
                "source":    "llm",
                "side":      side,
                "edge":      edge,
                "confidence": llm_result["confidence"],
                "weight":    0.40,
                "reasoning": llm_result.get("reasoning", ""),
            })

    # ── 2. Spread Arb ──
    arb = detect_spread_arb(market)
    if arb:
        signals.append({
            "source":    "spread_arb",
            "side":      arb["side"],
            "edge":      arb["gap"] * 0.75,
            "confidence": 0.85,
            "weight":    0.30,
        })

    # ── 3. Logical Arb ──
    over_arb = detect_overpriced_spread(market)
    if over_arb:
        signals.append({
            "source":    "logical_arb",
            "side":      over_arb["side"],
            "edge":      over_arb["edge"],
            "confidence": 0.80,
            "weight":    0.15,
        })

    # ── 4. Momentum ──
    mom_prob = get_momentum_signal(market.condition_id, market.yes_price)
    if mom_prob is not None:
        edge_yes = mom_prob - market.yes_price
        edge_no  = (1 - mom_prob) - market.no_price
        side     = "YES" if edge_yes >= edge_no else "NO"
        edge     = edge_yes if side == "YES" else edge_no
        if abs(edge) >= 0.04:
            vol_conf = min(0.72, 0.50 + (market.volume / 2_000_000) * 0.20)
            signals.append({
                "source":    "momentum",
                "side":      side,
                "edge":      edge,
                "confidence": vol_conf,
                "weight":    0.15,
            })

    if not signals:
        return None

    # ── Determine dominant side ──
    yes_sigs = [s for s in signals if s["side"] == "YES"]
    no_sigs  = [s for s in signals if s["side"] == "NO"]
    dominant = yes_sigs if sum(s["weight"] for s in yes_sigs) >= sum(s["weight"] for s in no_sigs) else no_sigs
    side     = "YES" if dominant is yes_sigs else "NO"

    if not dominant:
        return None

    # ── Signal quality gate ──
    if len(dominant) == 1:
        solo = dominant[0]
        # Solo signals need higher confidence
        if solo["confidence"] < 0.40:
            log_event(f"[SKIP] Solo {solo['source']} conf={solo['confidence']:.2f} too low for {market.question[:40]}")
            return None
        log_event(f"[SIGNAL] Solo {solo['source']} accepted conf={solo['confidence']:.2f}")
    else:
        # Multi-signal agreement → boost confidence by 6%
        for s in dominant:
            s["confidence"] = min(s["confidence"] + 0.06, 0.95)
        log_event(f"[SIGNAL] {len(dominant)}-signal agreement — conf boosted")

    # ── Weighted composite ──
    total_w  = sum(s["weight"] for s in dominant)
    avg_conf = sum(s["confidence"] * s["weight"] for s in dominant) / total_w
    avg_edge = sum(s["edge"]       * s["weight"] for s in dominant) / total_w

    # ── Composite quality gate — must be worth the trade ──
    composite = avg_conf * avg_edge
    if composite < 0.015:
        log_event(f"[SKIP] Composite {composite:.4f} below 0.015 for {market.question[:40]}")
        return None

    sources   = "+".join(sorted({s["source"] for s in dominant}))
    token_id  = market.yes_token_id if side == "YES" else market.no_token_id
    entry     = market.yes_price    if side == "YES" else market.no_price
    reasoning = next((s.get("reasoning", "") for s in dominant if s.get("reasoning")), "")

    signal = {
        "side":       side,
        "entry_price": entry,
        "token_id":   token_id,
        "edge":       round(avg_edge,  4),
        "confidence": round(avg_conf,  4),
        "reasoning":  reasoning,
        "source":     sources,
    }

    # Attach opportunity score for ranking
    signal["opportunity_score"] = score_opportunity(market, signal)

    log_event(
        f"[SIGNAL ✓] {side} {market.question[:42]} | "
        f"edge={avg_edge:.3f} conf={avg_conf:.2f} "
        f"score={signal['opportunity_score']:.3f} src={sources}"
    )
    return signal


# ──────────────────────────────────────────────────────────────────────────────
#  RISK MANAGER  — v4: checks actual entry price side, not just yes_price
# ──────────────────────────────────────────────────────────────────────────────
def risk_check(signal: dict, market: Market) -> tuple[bool, str]:
    edge       = signal["edge"]
    confidence = signal["confidence"]
    side       = signal["side"]
    entry      = signal["entry_price"]  # actual side we're entering

    open_pos = [p for p in state.positions if p.status == "OPEN"]

    if len(open_pos) >= MAX_OPEN_POSITIONS:
        return False, f"Max positions ({MAX_OPEN_POSITIONS}) reached"

    if edge < MIN_EDGE_PCT:
        return False, f"Edge {edge:.3f} below min {MIN_EDGE_PCT}"

    if confidence < 0.38:
        return False, f"Confidence {confidence:.2f} too low (need ≥ 0.38)"

    if market.volume < 20_000:
        return False, f"Volume ${market.volume:,.0f} below $20k floor"

    # ── KEY FIX v4: check the ENTRY price (the side we're actually buying) ──
    # not just market.yes_price — this was the alien trade bug
    if entry < ENTRY_MIN_PRICE or entry > ENTRY_MAX_PRICE:
        return False, f"Entry {entry:.3f} outside buy-low zone [{ENTRY_MIN_PRICE}–{ENTRY_MAX_PRICE}]"

    if market.condition_id in {p.condition_id for p in open_pos}:
        return False, "Already have position in this market"

    # ── Semantic dedup — no contradicting positions on same entity ──
    if _question_too_similar(market.question, open_pos):
        return False, f"Already have position on related market (entity dedup)"

    if LIVE_MODE:
        sz = position_size(edge, confidence)
        if sz < CLOB_MIN_ORDER_SIZE:
            return False, f"Size ${sz:.2f} below CLOB minimum ${CLOB_MIN_ORDER_SIZE}"

    return True, "OK"


def position_size(edge: float, confidence: float) -> float:
    """
    Kelly-fractional sizing scaled by conviction.
    Higher edge + confidence = larger bet (up to MAX_POSITION_PCT).
    """
    kelly = (edge * confidence) / max(1 - (edge * confidence), 0.01)
    # Use 30% Kelly — conservative but meaningful
    raw   = state.balance * kelly * 0.30
    # Scale: min $10, max 8% of balance
    sized = max(10.0, min(raw, state.balance * MAX_POSITION_PCT))
    return round(sized, 2)


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
            f"🟢 LIVE TRADE OPENED\n{market.question[:55]}\n"
            f"{signal['side']} @ {signal['entry_price']:.3f} | "
            f"edge={signal['edge']:.3f} conf={signal['confidence']:.2f} | ${size:.2f}\n"
            f"Signals: {signal.get('source','')} | {signal.get('reasoning','')}"
        )
    else:
        state.balance -= size

    shares = round(size / signal["entry_price"], 4)
    pos = Position(
        market_id=market.id,
        condition_id=market.condition_id,
        question=market.question,
        side=signal["side"],
        shares=shares,
        entry_price=signal["entry_price"],
        current_price=signal["entry_price"],
        timestamp=datetime.now(timezone.utc).isoformat(),
        claude_confidence=signal["confidence"],
        edge=signal["edge"],
        order_id=order_id,
        token_id=token_id,
        source=signal.get("source", "ai"),
        opportunity_score=signal.get("opportunity_score", 0.0),
    )
    state.positions.append(pos)
    state.trades_taken += 1

    mode = "LIVE" if LIVE_MODE else "PAPER"
    log_event(
        f"[{mode} OPEN] {signal['side']} {market.question[:50]} | "
        f"entry={signal['entry_price']:.3f} edge={signal['edge']:.3f} "
        f"conf={signal['confidence']:.2f} src={signal.get('source','?')} "
        f"size=${size:.2f} shares={shares}"
    )
    await append_trade_log({
        **asdict(pos), "event": "open", "mode": mode,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    return pos


# ──────────────────────────────────────────────────────────────────────────────
#  POSITION UPDATES & CLOSING  — v4 buy-low/sell-high exit logic
# ──────────────────────────────────────────────────────────────────────────────
async def update_positions(markets: list[Market]):
    market_lookup = {m.condition_id: m for m in markets}
    market_lookup.update({m.id: m for m in markets})

    for pos in state.positions:
        if pos.status != "OPEN":
            continue

        # ── Price discovery (multi-source) ──
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

        # ── v4 BUY-LOW / SELL-HIGH exit logic ──
        #
        # Take Profit:
        #   Exit when price reaches TAKE_PROFIT_TARGET (0.72) OR
        #   when we've gained at least TAKE_PROFIT_MIN_GAIN (0.22) over entry.
        #   This is the "sell high" part — lock in profits before reversal.
        #
        # Stop Loss:
        #   Dynamic: stop at 40% loss of entry price.
        #   E.g. entered at 0.30 → stop at 0.18 (loss of 0.12)
        #   But never below STOP_LOSS_FLOOR (0.07) regardless.
        #   This protects against catastrophic losses while giving room.

        take_profit = min(TAKE_PROFIT_TARGET, pos.entry_price + TAKE_PROFIT_MIN_GAIN + 0.05)
        # For cheap entries we want the price to rise significantly
        take_profit = max(take_profit, pos.entry_price + TAKE_PROFIT_MIN_GAIN)

        stop_loss = max(
            pos.entry_price * (1 - STOP_LOSS_PCT),  # 40% below entry
            STOP_LOSS_FLOOR,                          # never below 7¢
        )

        if pos.current_price >= take_profit:
            await _close_position(pos, pos.current_price, "TAKE_PROFIT")
        elif pos.current_price <= stop_loss:
            await _close_position(pos, pos.current_price, "STOP_LOSS")
        else:
            # ── Trailing profit protection ──
            # If position has gained > 30% of entry value and starts dropping,
            # move stop to break-even + 5% to lock in some profit
            gain_pct = (pos.current_price - pos.entry_price) / pos.entry_price
            if gain_pct > 0.30:
                breakeven_stop = pos.entry_price * 1.05  # +5% buffer
                if pos.current_price < breakeven_stop and pos.pnl > 0:
                    log_event(f"[TRAILING] Locking profit on {pos.question[:40]} — closing near breakeven+5%")
                    await _close_position(pos, pos.current_price, "TRAILING_STOP")


async def _close_position(pos: Position, price: float, reason: str):
    pos.pnl         = round((price - pos.entry_price) * pos.shares, 4)
    pos.close_price = price
    pos.close_time  = datetime.now(timezone.utc).isoformat()
    pos.status      = "CLOSED"

    # Return capital + profit/loss
    proceeds         = price * pos.shares
    state.balance    = round(state.balance + proceeds, 2)
    state.total_pnl  = round(state.total_pnl + pos.pnl, 4)
    state.closed_trades.append(asdict(pos))

    # ── Cooldown — prevent re-entering losing markets ──
    now_ts = datetime.now(timezone.utc).timestamp()
    if reason == "STOP_LOSS":
        _recently_closed[pos.condition_id] = now_ts + 82800   # 24hr ban
        log_event(f"[COOLDOWN 24H] {pos.question[:40]} — banned 24hr after stop loss")
    elif reason == "TRAILING_STOP":
        _recently_closed[pos.condition_id] = now_ts + 7200    # 2hr ban
    else:
        _recently_closed[pos.condition_id] = now_ts           # standard 1hr

    mode   = "LIVE" if (LIVE_MODE and pos.order_id) else "PAPER"
    emoji  = "🟢" if pos.pnl >= 0 else "🔴"
    log_event(
        f"[{mode} CLOSE/{reason}] {emoji} {pos.question[:45]} | "
        f"PnL=${pos.pnl:+.2f} entry={pos.entry_price:.3f} → close={price:.3f} | "
        f"shares={pos.shares}"
    )

    if abs(pos.pnl) > 2 or reason == "STOP_LOSS":
        await send_alert(
            f"{emoji} {reason}\n{pos.question[:55]}\n"
            f"PnL: ${pos.pnl:+.2f} | {pos.side} @ {pos.entry_price:.3f}→{price:.3f} | {mode}"
        )

    await append_trade_log({
        **asdict(pos), "event": "close", "reason": reason,
        "mode": mode, "ts": pos.close_time,
    })
    await save_state()


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN SCAN CYCLE  — v4: rank by opportunity score, pick top N
# ──────────────────────────────────────────────────────────────────────────────
async def scan_cycle():
    state.scan_count += 1
    log_event(f"── Scan #{state.scan_count} [{'LIVE' if LIVE_MODE else 'PAPER'}] " + "─" * 30)

    markets = await fetch_markets(200)
    await update_positions(markets)

    open_cids  = {p.condition_id for p in state.positions if p.status == "OPEN"}
    now_ts     = datetime.now(timezone.utc).timestamp()
    on_cooldown = {cid for cid, ts in _recently_closed.items() if now_ts - ts < COOLDOWN_SECONDS}
    if on_cooldown:
        log_event(f"[COOLDOWN] {len(on_cooldown)} market(s) skipped")

    open_positions_list = [p for p in state.positions if p.status == "OPEN"]

    # ── Filter candidates ──
    # Core buy-low filter: we ONLY look at markets with at least one side priced
    # in the ENTRY_MIN_PRICE–ENTRY_MAX_PRICE buy zone (0.12–0.45)
    candidates = []
    for m in markets:
        if m.condition_id in open_cids:
            continue
        if m.condition_id in on_cooldown:
            continue
        if m.volume < 20_000:
            continue
        if not question_allowed(m.question):
            continue
        if _question_too_similar(m.question, open_positions_list):
            continue

        # Buy-low filter: at least one side must be in the cheap zone
        yes_in_zone = ENTRY_MIN_PRICE <= m.yes_price <= ENTRY_MAX_PRICE
        no_in_zone  = ENTRY_MIN_PRICE <= m.no_price  <= ENTRY_MAX_PRICE
        if not (yes_in_zone or no_in_zone):
            continue

        candidates.append(m)

    # Sort by volume first to ensure we analyse the most liquid markets
    candidates.sort(key=lambda m: m.volume, reverse=True)
    candidates = candidates[:30]  # analyse top 30

    log_event(f"[SCAN] {len(candidates)} candidates in buy zone from {len(markets)} markets")

    if not candidates:
        log_event("[SCAN] No candidates in buy zone — all markets outside 0.12–0.45 range")
        await save_state()
        return

    # ── Analyse all candidates and collect signals ──
    ranked_signals: list[tuple[Market, dict]] = []

    for market in candidates:
        state.signals_analyzed += 1
        signal = await analyse_market(market)
        if not signal:
            continue

        allow, reason = risk_check(signal, market)
        if not allow:
            log_event(
                f"[RISK SKIP] {market.question[:45]} — {reason} "
                f"(edge={signal['edge']:.3f} conf={signal['confidence']:.2f})"
            )
            continue

        ranked_signals.append((market, signal))
        await asyncio.sleep(1)  # rate limit LLM calls slightly

    # ── Rank by opportunity score and trade the BEST ones ──
    ranked_signals.sort(key=lambda x: x[1]["opportunity_score"], reverse=True)

    slots_available = MAX_OPEN_POSITIONS - len(open_positions_list)
    log_event(f"[SCAN] {len(ranked_signals)} signals passed risk check | {slots_available} slots available")

    trades_this_scan = 0
    for market, signal in ranked_signals:
        if trades_this_scan >= slots_available:
            log_event(f"[SCAN] Slots full — best remaining signal was score={signal['opportunity_score']:.3f}")
            break
        # Re-check dedup after each trade (in case we just opened a related position)
        current_open = [p for p in state.positions if p.status == "OPEN"]
        if _question_too_similar(market.question, current_open):
            log_event(f"[SKIP late dedup] {market.question[:45]}")
            continue
        if len(current_open) >= MAX_OPEN_POSITIONS:
            break

        pos = await open_position(market, signal)
        if pos:
            trades_this_scan += 1
            await asyncio.sleep(2)

    if trades_this_scan == 0 and ranked_signals:
        log_event(f"[SCAN] No new trades — {len(ranked_signals)} signals rejected by risk/dedup")
    elif trades_this_scan == 0:
        log_event("[SCAN] No signals passed all filters this cycle")

    prune_memory()
    await save_state()

    open_count = sum(1 for p in state.positions if p.status == "OPEN")
    wins  = sum(1 for t in state.closed_trades if t.get("pnl", 0) > 0)
    total = len(state.closed_trades)
    wr    = f"{wins/total*100:.0f}%" if total > 0 else "—"
    log_event(
        f"Balance: ${state.balance:.2f} | PnL: ${state.total_pnl:+.2f} | "
        f"Open: {open_count} | Trades: {state.trades_taken} | WinRate: {wr}"
    )


# ──────────────────────────────────────────────────────────────────────────────
#  WEB DASHBOARD  — v4: shows opportunity scores, win rate, buy-low strategy
# ──────────────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PolyBot v4 — Buy Low / Sell High</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#070b10;--s:#0d1520;--b:#172030;--t:#c8d8f0;--m:#4a5878;
      --a:#3b82f6;--g:#22c55e;--r:#ef4444;--y:#f59e0b;--p:#a855f7;--o:#f97316}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--t);min-height:100vh;padding:24px 20px}
.mono{font-family:'JetBrains Mono',monospace}
.hdr{display:flex;align-items:center;gap:10px;margin-bottom:4px;flex-wrap:wrap}
h1{font-size:20px;font-weight:700;color:#fff}
.tagline{font-size:11px;color:var(--y);font-weight:600;padding:2px 8px;background:rgba(245,158,11,.12);border-radius:4px}
.badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600}
.badge.live{background:#14532d;color:#86efac}.badge.paper{background:#1e3a5f;color:#93c5fd}
.badge.redis{background:#3b1f6e;color:#c4b5fd}.badge.json{background:#1c1917;color:#d6d3d1}
.dot{width:6px;height:6px;border-radius:50%;background:currentColor;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.sub{font-size:11px;color:var(--m);margin-bottom:18px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px;margin-bottom:16px}
.stat{background:var(--s);border:1px solid var(--b);border-radius:8px;padding:12px 14px}
.sl{font-size:10px;color:var(--m);text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px}
.sv{font-size:20px;font-weight:600;color:#fff;font-family:'JetBrains Mono',monospace}
.sv.g{color:var(--g)}.sv.r{color:var(--r)}.sv.b{color:var(--a)}.sv.p{color:var(--p)}.sv.o{color:var(--o)}
/* Strategy info box */
.strategy{background:rgba(249,115,22,.06);border:1px solid rgba(249,115,22,.25);border-radius:8px;padding:12px 16px;margin-bottom:16px;font-size:12px}
.strategy strong{color:var(--o)}
.strategy .params{display:flex;gap:16px;margin-top:8px;flex-wrap:wrap}
.strategy .param{color:var(--m)}.strategy .pval{color:#fff;font-family:'JetBrains Mono',monospace}
/* Integrity */
.integrity{background:var(--s);border:1px solid var(--b);border-radius:8px;padding:14px 16px;margin-bottom:16px}
.integrity.ok{border-color:#166534}.integrity.warn{border-color:#854d0e}
.int-title{font-size:10px;font-weight:600;color:var(--m);text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px}
.int-row{display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px}
.int-label{color:var(--m)}.int-val{font-family:'JetBrains Mono',monospace;color:#fff}
.int-status{font-size:11px;font-weight:600;margin-top:8px;padding:4px 10px;border-radius:4px;display:inline-block}
.int-status.ok{background:#14532d;color:#86efac}.int-status.warn{background:#451a03;color:#fde68a}.int-status.na{background:#1c1917;color:#9ca3af}
/* Tabs */
.tabs{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap}
.tab{background:var(--s);border:1px solid var(--b);border-radius:5px;padding:5px 12px;font-size:11px;cursor:pointer;color:var(--m);transition:all .15s}
.tab:hover{color:var(--t)}.tab.active{background:var(--a);border-color:var(--a);color:#fff}
.tw{background:var(--s);border:1px solid var(--b);border-radius:8px;overflow:hidden;margin-bottom:16px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:9px 11px;color:var(--m);font-weight:500;border-bottom:1px solid var(--b);font-size:10px;text-transform:uppercase;letter-spacing:.06em}
td{padding:8px 11px;border-bottom:1px solid rgba(23,32,48,.5)}
tr:last-child td{border-bottom:none}
tr:hover td{background:rgba(59,130,246,.04)}
.pill{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600;font-family:'JetBrains Mono',monospace}
.pill.yes{background:#14532d;color:#86efac}.pill.no{background:#450a0a;color:#fca5a5}
.pill.ai{background:#1c1917;color:#d6d3d1}.pill.arb{background:#1e1b4b;color:#a5b4fc}
.pill.mom{background:#172030;color:#7dd3fc}.pill.multi{background:#2d1b69;color:#c4b5fd}
.pp{color:var(--g)}.pn{color:var(--r)}
.live-dot{color:var(--g);font-size:11px}.paper-dot{color:var(--a);font-size:11px}
.log{background:var(--s);border:1px solid var(--b);border-radius:8px;padding:12px;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--m);max-height:300px;overflow-y:auto;line-height:1.8}
.le.err{color:#fca5a5}.le.ok{color:#86efac}.le.ai{color:#fde68a}.le.skip{color:#6b7280}
.tb{display:flex;gap:8px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
.btn{background:var(--a);color:#fff;border:none;border-radius:5px;padding:6px 14px;font-size:12px;cursor:pointer;font-family:inherit;font-weight:500}
.btn:hover{background:#1d4ed8}
.upd{font-size:11px;color:var(--m)}.empty{padding:20px;color:var(--m);font-size:12px;text-align:center}
.hidden{display:none}
/* Score bar */
.score-bar{display:flex;align-items:center;gap:6px}
.score-fill{height:4px;border-radius:2px;background:var(--a)}
.wr-good{color:var(--g)}.wr-med{color:var(--y)}.wr-bad{color:var(--r)}
</style>
</head>
<body>
<div class="hdr">
  <h1>PolyBot v4</h1>
  <span class="tagline">BUY LOW · SELL HIGH</span>
  <span class="badge paper" id="mode-badge"><span class="dot"></span>PAPER</span>
  <span class="badge json" id="store-badge">JSON</span>
</div>
<p class="sub">Enters cheap positions (12¢–45¢) · Exits at 72¢+ target · Dynamic stop-loss · No sports/meme · 4-signal AI · auto-refresh 20s</p>

<div class="strategy">
  <strong>Strategy: Buy Low / Sell High on Prediction Markets</strong>
  <div class="params">
    <span class="param">Entry zone: <span class="pval">$0.12 – $0.45</span></span>
    <span class="param">Take profit: <span class="pval">≥$0.72 or +$0.22 gain</span></span>
    <span class="param">Stop loss: <span class="pval">–40% of entry</span></span>
    <span class="param">Max positions: <span class="pval" id="max-pos">5</span></span>
    <span class="param">Min edge: <span class="pval">4%</span></span>
  </div>
</div>

<div class="tb">
  <button class="btn" onclick="loadAll()">↻ Refresh</button>
  <span class="upd" id="upd">Loading…</span>
</div>

<div class="stats" id="stats"></div>

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
  if(src.includes('arb')&&src.includes('llm'))return'<span class="pill multi">MULTI</span>';
  if(src.includes('arb'))return'<span class="pill arb">ARB</span>';
  if(src.includes('momentum'))return'<span class="pill mom">MOM</span>';
  return'<span class="pill ai">AI</span>';
}

function scoreBar(score){
  const pct=Math.min(100,Math.round((score||0)*500));
  return`<div class="score-bar"><div class="score-fill" style="width:${pct}px;max-width:60px"></div><span style="font-size:10px;color:#6b7280">${f(score||0,3)}</span></div>`;
}

function wrColor(wr){
  if(wr==='—')return'';
  const n=parseFloat(wr);
  if(n>=60)return'wr-good';if(n>=40)return'wr-med';return'wr-bad';
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
  await Promise.all([loadState(),loadLedger()]);
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
    const cl=d.closed_trades||[];
    const wins=cl.filter(t=>(t.pnl||0)>0).length;
    const total=cl.length;
    const wr=total>0?Math.round(wins/total*100)+'%':'—';
    const wrCls=wrColor(wr);

    document.getElementById('stats').innerHTML=`
      <div class="stat"><div class="sl">Balance</div><div class="sv mono">$${f(d.balance)}</div></div>
      <div class="stat"><div class="sl">Total PnL</div><div class="sv ${pc(pnl)} mono">${ps(pnl)}</div></div>
      <div class="stat"><div class="sl">Win Rate</div><div class="sv ${wrCls}">${wr}</div></div>
      <div class="stat"><div class="sl">Trades</div><div class="sv b">${d.trades_taken||0}</div></div>
      <div class="stat"><div class="sl">Open</div><div class="sv">${(d.open_positions||[]).length}</div></div>
      <div class="stat"><div class="sl">Scans</div><div class="sv">${d.scan_count||0}</div></div>
      <div class="stat"><div class="sl">Signals</div><div class="sv">${d.signals_analyzed||0}</div></div>`;

    document.getElementById('int-running').textContent='$'+f(pnl);

    const ops=d.open_positions||[];
    document.getElementById('pos-tbl').innerHTML=ops.length===0
      ?'<div class="empty">No open positions — bot is scanning for buy-low opportunities</div>'
      :`<table>
        <thead><tr><th>Market</th><th>Side</th><th>Signals</th><th>Entry</th><th>Now</th><th>Target</th><th>PnL</th><th>Score</th><th>Conf</th><th>Mode</th></tr></thead>
        <tbody>${ops.map(p=>{
          const tp=Math.min(0.72,p.entry_price+(0.27)).toFixed(3);
          const gainPct=p.entry_price?((p.current_price-p.entry_price)/p.entry_price*100).toFixed(1):'0';
          return`<tr>
            <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${p.question}">${p.question.slice(0,48)}${p.question.length>48?'…':''}</td>
            <td><span class="pill ${p.side.toLowerCase()}">${p.side}</span></td>
            <td>${srcPill(p.source)}</td>
            <td class="mono">${f(p.entry_price,3)}</td>
            <td class="mono" style="color:${p.current_price>p.entry_price?'#22c55e':p.current_price<p.entry_price?'#ef4444':'#c8d8f0'}">${f(p.current_price,3)}</td>
            <td class="mono" style="color:#f59e0b">${tp}</td>
            <td class="mono ${pc(p.pnl)}">${ps(p.pnl)} <span style="font-size:9px;opacity:.6">${gainPct}%</span></td>
            <td>${scoreBar(p.opportunity_score)}</td>
            <td>${p.claude_confidence?f(p.claude_confidence*100,0)+'%':'—'}</td>
            <td>${p.order_id?'<span class="live-dot">● LIVE</span>':'<span class="paper-dot">◌ PAPER</span>'}</td>
          </tr>`;
        }).join('')}</tbody></table>`;

    const clsd=(d.closed_trades||[]).slice(-30).reverse();
    document.getElementById('cls-tbl').innerHTML=clsd.length===0
      ?'<div class="empty">No closed trades yet</div>'
      :`<table>
        <thead><tr><th>Market</th><th>Side</th><th>Entry</th><th>Close</th><th>PnL</th><th>Return%</th><th>Reason</th><th>Time</th></tr></thead>
        <tbody>${clsd.map(p=>{
          const ret=p.entry_price?((p.close_price-p.entry_price)/p.entry_price*100).toFixed(1):'0';
          return`<tr>
            <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${p.question}">${p.question.slice(0,48)}${p.question.length>48?'…':''}</td>
            <td><span class="pill ${p.side.toLowerCase()}">${p.side}</span></td>
            <td class="mono">${f(p.entry_price,3)}</td>
            <td class="mono">${f(p.close_price||0,3)}</td>
            <td class="mono ${pc(p.pnl)}">${ps(p.pnl)}</td>
            <td class="${+ret>=0?'pp':'pn'}">${ret}%</td>
            <td style="font-size:10px">${p.reason||'—'}</td>
            <td style="font-size:10px">${p.close_time?new Date(p.close_time).toLocaleTimeString():'-'}</td>
          </tr>`;
        }).join('')}</tbody></table>`;

    const ll=(d.log||[]).slice().reverse();
    document.getElementById('log').innerHTML=ll.map(l=>{
      let c='le';
      if(l.includes('ERROR')||l.includes('FAIL')||l.includes('STOP_LOSS'))c+=' err';
      else if(l.includes('TAKE_PROFIT')||l.includes('OPEN')||l.includes('✓'))c+=' ok';
      else if(l.includes('LLM')||l.includes('SIGNAL'))c+=' ai';
      else if(l.includes('SKIP')||l.includes('COOLDOWN'))c+=' skip';
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
    const running=+(document.getElementById('int-running').textContent.replace(/[^0-9.-]/g,''))||0;
    document.getElementById('int-ledger').textContent=ledgerTotal!=null?'$'+f(ledgerTotal):'Not available (no Redis)';
    const box=document.getElementById('integrity-box');
    const statusEl=document.getElementById('int-status');
    if(ledgerTotal==null){
      document.getElementById('int-drift').textContent='N/A';
      statusEl.textContent='Redis not connected';statusEl.className='int-status na';box.className='integrity';
    } else {
      const drift=Math.abs(running-ledgerTotal);
      document.getElementById('int-drift').textContent='$'+f(drift);
      if(drift<0.01){statusEl.textContent='✓ PnL verified — no drift';statusEl.className='int-status ok';box.className='integrity ok';}
      else{statusEl.textContent=`⚠ Drift $${f(drift)} — check logs`;statusEl.className='int-status warn';box.className='integrity warn';}
    }
    document.getElementById('ledger-tbl').innerHTML=ledger.length===0
      ?'<div class="empty">No ledger entries yet</div>'
      :`<table>
        <thead><tr><th>Time</th><th>Market</th><th>Side</th><th>Entry</th><th>Close</th><th>PnL</th><th>Reason</th><th>Mode</th></tr></thead>
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
  }catch(e){console.warn('Ledger load failed:',e.message);}
}

loadAll();
setInterval(loadAll,20000);
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────────────
#  WEB SERVER HANDLERS
# ──────────────────────────────────────────────────────────────────────────────
async def _handle_root(request):
    return aio_web.Response(text=DASHBOARD_HTML, content_type="text/html")


async def _handle_trades(request):
    r       = await get_redis()
    storage = "Redis" if (r is not None and _redis_available) else "JSON"
    cl      = state.closed_trades
    wins    = sum(1 for t in cl if t.get("pnl", 0) > 0)
    total   = len(cl)
    return aio_web.json_response({
        "balance":          state.balance,
        "total_pnl":        state.total_pnl,
        "scan_count":       state.scan_count,
        "signals_analyzed": state.signals_analyzed,
        "trades_taken":     state.trades_taken,
        "start_time":       state.start_time,
        "mode":             "LIVE" if LIVE_MODE else "PAPER",
        "storage":          storage,
        "win_rate":         round(wins / total, 3) if total > 0 else 0,
        "wins":             wins,
        "losses":           total - wins,
        "open_positions":   [asdict(p) for p in state.positions if p.status == "OPEN"],
        "closed_trades":    state.closed_trades[-100:],
        "log":              state.log[-200:],
        "updated_at":       datetime.now(timezone.utc).isoformat(),
    })


async def _handle_ledger(request):
    limit        = int(request.rel_url.query.get("limit", 100))
    ledger_total = await redis_get_ledger_total()
    entries      = await redis_get_ledger_entries(limit)
    return aio_web.json_response({
        "ledger_total": ledger_total,
        "entry_count":  len(entries),
        "entries":      entries,
    })


async def _handle_health(request):
    r        = await get_redis()
    redis_ok = r is not None and _redis_available
    cl       = state.closed_trades
    wins     = sum(1 for t in cl if t.get("pnl", 0) > 0)
    total    = len(cl)
    return aio_web.json_response({
        "status":         "ok",
        "mode":           "LIVE" if LIVE_MODE else "PAPER",
        "storage":        "redis" if redis_ok else "json",
        "scans":          state.scan_count,
        "open_positions": sum(1 for p in state.positions if p.status == "OPEN"),
        "balance":        round(state.balance, 2),
        "total_pnl":      round(state.total_pnl, 4),
        "trades":         state.trades_taken,
        "win_rate":       round(wins / total, 3) if total > 0 else 0,
        "strategy":       "buy_low_sell_high",
        "entry_zone":     f"{ENTRY_MIN_PRICE}–{ENTRY_MAX_PRICE}",
        "take_profit":    TAKE_PROFIT_TARGET,
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
#  MAIN BOOT
# ──────────────────────────────────────────────────────────────────────────────
async def bot_loop():
    await start_web()
    asyncio.create_task(self_ping())
    asyncio.create_task(ws_price_feed())

    load_saved_state()

    # Ledger reset support
    if os.getenv("RESET_LEDGER", "false").lower() == "true":
        r = await get_redis()
        if r:
            await r.delete(RK_PNL_LEDGER)
            await redis_append_pnl_ledger({
                "trade_id":    "ledger_reset",
                "question":    "Ledger reset correction",
                "side":        "RESET",
                "pnl":         state.total_pnl,
                "entry_price": 0, "close_price": 0, "shares": 0,
                "reason":      "LEDGER_RESET", "mode": "PAPER",
            })
            log_event(f"[REDIS] Ledger reset — resynced to ${state.total_pnl:+.2f}")

    mode_str  = "🔴 LIVE TRADING — REAL MONEY" if LIVE_MODE else "📄 PAPER TRADING (safe)"
    r         = await get_redis()
    store_str = f"Redis ({REDIS_URL[:25]}…)" if (r is not None and _redis_available) else "JSON file"

    cl    = state.closed_trades
    wins  = sum(1 for t in cl if t.get("pnl", 0) > 0)
    total = len(cl)
    wr    = f"{wins/total*100:.0f}%" if total > 0 else "—"

    log_event(f"PolyBot v4 — BUY LOW / SELL HIGH — {mode_str}")
    log_event(f"State storage: {store_str}")
    log_event(f"LLM: Replicate → {REPLICATE_MODEL}")
    log_event(f"Strategy: Entry $0.12–$0.45 | Take-profit $0.72+ | Stop-loss –40%")
    log_event(f"Signals: Spread-Arb(0.30) + Logical-Arb(0.15) + Momentum(0.15) + LLM(0.40)")
    log_event(f"Filters: No sports, No memes, Semantic dedup, Entity-level dedup")
    log_event(f"Max positions: {MAX_OPEN_POSITIONS} | Min edge: {MIN_EDGE_PCT} | Min volume: $20k")
    log_event(f"Balance: ${state.balance:.2f} | PnL: ${state.total_pnl:+.2f} | Trades: {state.trades_taken} | Win rate: {wr}")
    log_event("─" * 60)

    if LIVE_MODE:
        live_bal = await clob_get_balance()
        if live_bal > 0:
            state.balance = live_bal
            log_event(f"[LIVE] Chain balance synced: ${live_bal:.2f} USDC")
        await send_alert(
            f"🤖 PolyBot v4 LIVE started\n"
            f"Strategy: Buy $0.12–$0.45 → Sell $0.72+\n"
            f"Balance: ${state.balance:.2f} | PnL: ${state.total_pnl:+.2f}\n"
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