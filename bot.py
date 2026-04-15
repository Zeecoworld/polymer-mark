import asyncio
import json
import random
import httpx
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from aiohttp import web as aio_web


from dotenv import load_dotenv


# Load environment variables from the .env file (if present)
load_dotenv()

# ==============================================================================
#  CONFIG
# ==============================================================================
REPLICATE_API_KEY  = os.getenv("REPLICATE_API_KEY", "YOUR_r8_TOKEN_HERE")
REPLICATE_MODEL    = "meta/meta-llama-3-70b-instruct"
REPLICATE_API_URL  = "https://api.replicate.com/v1/models/{model}/predictions"

PAPER_BALANCE_USDC  = 1000.0
GAMMA_API           = "https://gamma-api.polymarket.com"
SCAN_INTERVAL       = 60
MAX_POSITION_PCT    = 0.04
MIN_EDGE_PCT        = 0.05
MAX_OPEN_POSITIONS  = 3
LOG_FILE            = "paper_trades.json"

# ==============================================================================
#  STARTUP GUARD  — catches missing API key before bot wastes any cycles
# ==============================================================================
if REPLICATE_API_KEY.startswith("YOUR_"):
    raise RuntimeError(
        "REPLICATE_API_KEY is not set. "
        "Add it as an environment variable in your Render dashboard."
    )

# ==============================================================================
#  FREE RSS NEWS FEEDS
# ==============================================================================
RSS_FEEDS = {
    "sports": [
        "https://www.espn.com/espn/rss/news",
        "https://feeds.bbci.co.uk/sport/rss.xml",
        "https://www.skysports.com/rss/12040",
        "https://news.google.com/rss/search?q=sports&hl=en",
    ],
    "crypto": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://news.google.com/rss/search?q=bitcoin+crypto&hl=en",
    ],
    "general": [
        "https://feeds.reuters.com/reuters/topNews",
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://news.google.com/rss/headlines?hl=en",
    ],
}

_news_cache: dict = {}
_news_cache_time: dict = {}
NEWS_CACHE_TTL = 300

# ==============================================================================
#  SINGLE SHARED HTTP CLIENT with connection limits
# ==============================================================================
_http: Optional[httpx.AsyncClient] = None

async def get_http() -> httpx.AsyncClient:
    global _http
    if _http is None or _http.is_closed:
        _http = httpx.AsyncClient(
            timeout=15,
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=5,
                max_keepalive_connections=2
            )
        )
    return _http

# ==============================================================================
#  RSS NEWS — max 2 feeds per scan
# ==============================================================================
async def fetch_rss(url: str) -> list[str]:
    try:
        http = await get_http()
        r = await http.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.text)
        headlines = []
        for item in root.iter("item"):
            title = item.find("title")
            if title is not None and title.text:
                headlines.append(title.text.strip())
        for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title")
            if title is not None and title.text:
                headlines.append(title.text.strip())
        return headlines[:10]
    except Exception:
        return []


async def get_news_for_market(market_question: str, category: str) -> str:
    now = datetime.now(timezone.utc).timestamp()
    cat = category.lower()

    if "sport" in cat or any(w in market_question.lower() for w in
       ["nba", "nfl", "fifa", "epl", "soccer", "football", "basketball",
        "tennis", "golf", "baseball", "nhl", "mls", "ufc", "boxing"]):
        feed_keys = ["sports", "general"]
    elif any(w in market_question.lower() for w in
             ["bitcoin", "btc", "eth", "crypto", "solana", "coinbase"]):
        feed_keys = ["crypto", "general"]
    else:
        feed_keys = ["general", "sports", "crypto"]

    all_headlines = []
    for key in feed_keys[:1]:
        for url in RSS_FEEDS.get(key, [])[:2]:
            cache_key = url
            if (cache_key in _news_cache and
                    now - _news_cache_time.get(cache_key, 0) < NEWS_CACHE_TTL):
                all_headlines.extend(_news_cache[cache_key])
            else:
                headlines = await fetch_rss(url)
                _news_cache[cache_key] = headlines
                _news_cache_time[cache_key] = now
                all_headlines.extend(headlines)

    if not all_headlines:
        return "No recent news found."

    keywords = [w.lower() for w in market_question.replace("?", "").split()
                if len(w) > 3]

    def relevance(h: str) -> int:
        h_lower = h.lower()
        return sum(1 for kw in keywords if kw in h_lower)

    scored = sorted(set(all_headlines), key=relevance, reverse=True)
    top    = [h for h in scored if relevance(h) > 0][:5]
    if not top:
        top = scored[:3]

    return " | ".join(top) if top else "No relevant news found."

# ==============================================================================
#  AIOHTTP WEB SERVER — dashboard + API routes
# ==============================================================================
async def _handle_root(request):
    if os.path.exists("dashboard.html"):
        return aio_web.FileResponse("dashboard.html")
    # Inline fallback dashboard if no dashboard.html file present
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Polymarket Bot Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e2e8f0; min-height: 100vh; padding: 24px; }
  h1 { font-size: 22px; font-weight: 600; margin-bottom: 6px; color: #fff; }
  .subtitle { font-size: 13px; color: #64748b; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .card { background: #1e2130; border: 1px solid #2d3348; border-radius: 10px; padding: 16px; }
  .card-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
  .card-value { font-size: 24px; font-weight: 600; color: #fff; }
  .card-value.green { color: #22c55e; }
  .card-value.red { color: #ef4444; }
  .section-title { font-size: 14px; font-weight: 600; color: #94a3b8; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 8px 10px; color: #64748b; font-weight: 500; border-bottom: 1px solid #2d3348; }
  td { padding: 8px 10px; border-bottom: 1px solid #1e2130; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }
  .badge.yes { background: #14532d; color: #86efac; }
  .badge.no  { background: #450a0a; color: #fca5a5; }
  .badge.open { background: #1e3a5f; color: #93c5fd; }
  .log-box { background: #1e2130; border: 1px solid #2d3348; border-radius: 10px;
             padding: 14px; font-family: monospace; font-size: 12px; color: #94a3b8;
             max-height: 280px; overflow-y: auto; }
  .log-line { margin-bottom: 3px; line-height: 1.5; }
  .pnl-pos { color: #22c55e; }
  .pnl-neg { color: #ef4444; }
  .refresh-btn { background: #2563eb; color: #fff; border: none; border-radius: 6px;
                 padding: 8px 16px; font-size: 13px; cursor: pointer; margin-bottom: 20px; }
  .refresh-btn:hover { background: #1d4ed8; }
  .updated { font-size: 11px; color: #475569; margin-bottom: 16px; }
  #positions-table, #trades-table { background: #1e2130; border: 1px solid #2d3348; border-radius: 10px; overflow: hidden; margin-bottom: 24px; }
</style>
</head>
<body>
<h1>Polymarket AI Bot</h1>
<p class="subtitle">Paper trading dashboard — auto-refreshes every 30s</p>
<button class="refresh-btn" onclick="load()">Refresh Now</button>
<p class="updated" id="updated">Loading...</p>

<div class="grid" id="stats"></div>

<div class="section-title">Open Positions</div>
<div id="positions-table"></div>

<div class="section-title">Closed Trades</div>
<div id="trades-table"></div>

<div class="section-title" style="margin-top:24px">Bot Log</div>
<div class="log-box" id="log"></div>

<script>
async function load() {
  try {
    const r = await fetch('/paper_trades.json?t=' + Date.now());
    const d = await r.json();

    document.getElementById('updated').textContent =
      'Last updated: ' + new Date(d.updated_at).toLocaleTimeString();

    const pnlClass = d.total_pnl >= 0 ? 'green' : 'red';
    const pnlSign  = d.total_pnl >= 0 ? '+' : '';
    document.getElementById('stats').innerHTML = `
      <div class="card"><div class="card-label">Balance</div>
        <div class="card-value">$${d.balance.toFixed(2)}</div></div>
      <div class="card"><div class="card-label">Total PnL</div>
        <div class="card-value ${pnlClass}">${pnlSign}$${d.total_pnl.toFixed(2)}</div></div>
      <div class="card"><div class="card-label">Scans</div>
        <div class="card-value">${d.scan_count}</div></div>
      <div class="card"><div class="card-label">Signals</div>
        <div class="card-value">${d.signals_analyzed}</div></div>
      <div class="card"><div class="card-label">Trades</div>
        <div class="card-value">${d.trades_taken}</div></div>
      <div class="card"><div class="card-label">Open</div>
        <div class="card-value">${d.open_positions.length}</div></div>
    `;

    const posHtml = d.open_positions.length === 0
      ? '<p style="padding:16px;color:#64748b;font-size:13px">No open positions</p>'
      : `<table><thead><tr>
           <th>Market</th><th>Side</th><th>Entry</th>
           <th>Current</th><th>PnL</th><th>Confidence</th>
         </tr></thead><tbody>${d.open_positions.map(p => {
           const pnlCls = p.pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
           const pnlStr = (p.pnl >= 0 ? '+' : '') + '$' + p.pnl.toFixed(2);
           return `<tr>
             <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                 title="${p.question}">${p.question.slice(0,45)}${p.question.length>45?'...':''}</td>
             <td><span class="badge ${p.side.toLowerCase()}">${p.side}</span></td>
             <td>${p.entry_price.toFixed(3)}</td>
             <td>${p.current_price.toFixed(3)}</td>
             <td class="${pnlCls}">${pnlStr}</td>
             <td>${(p.claude_confidence * 100).toFixed(0)}%</td>
           </tr>`;
         }).join('')}</tbody></table>`;
    document.getElementById('positions-table').innerHTML = posHtml;

    const recent = (d.closed_trades || []).slice(-20).reverse();
    const trdHtml = recent.length === 0
      ? '<p style="padding:16px;color:#64748b;font-size:13px">No closed trades yet</p>'
      : `<table><thead><tr>
           <th>Market</th><th>Side</th><th>Entry</th>
           <th>Close</th><th>PnL</th><th>Closed At</th>
         </tr></thead><tbody>${recent.map(p => {
           const pnlCls = p.pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
           const pnlStr = (p.pnl >= 0 ? '+' : '') + '$' + p.pnl.toFixed(2);
           const closeTime = p.close_time ? new Date(p.close_time).toLocaleTimeString() : '-';
           return `<tr>
             <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                 title="${p.question}">${p.question.slice(0,45)}${p.question.length>45?'...':''}</td>
             <td><span class="badge ${p.side.toLowerCase()}">${p.side}</span></td>
             <td>${p.entry_price.toFixed(3)}</td>
             <td>${(p.close_price||0).toFixed(3)}</td>
             <td class="${pnlCls}">${pnlStr}</td>
             <td>${closeTime}</td>
           </tr>`;
         }).join('')}</tbody></table>`;
    document.getElementById('trades-table').innerHTML = trdHtml;

    const logLines = (d.log || []).slice().reverse();
    document.getElementById('log').innerHTML =
      logLines.map(l => `<div class="log-line">${l}</div>`).join('') ||
      '<div class="log-line">No log entries yet.</div>';

  } catch(e) {
    document.getElementById('updated').textContent = 'Error loading data: ' + e.message;
  }
}
load();
setInterval(load, 30000);
</script>
</body>
</html>"""
    return aio_web.Response(text=html, content_type="text/html")


async def _handle_trades(request):
    if os.path.exists(LOG_FILE):
        return aio_web.FileResponse(LOG_FILE)
    return aio_web.json_response({
        "balance":          PAPER_BALANCE_USDC,
        "total_pnl":        0.0,
        "scan_count":       0,
        "signals_analyzed": 0,
        "trades_taken":     0,
        "start_time":       datetime.now(timezone.utc).isoformat(),
        "open_positions":   [],
        "closed_trades":    [],
        "log":              ["Bot starting up..."],
        "updated_at":       datetime.now(timezone.utc).isoformat(),
    })


async def _handle_health(request):
    open_count = sum(1 for p in state.positions if p.status == "OPEN")
    return aio_web.json_response({
        "status":           "ok",
        "bot":              "running",
        "scans":            state.scan_count,
        "open_positions":   open_count,
        "balance":          round(state.balance, 2),
        "total_pnl":        round(state.total_pnl, 2),
    })


async def start_web():
    app = aio_web.Application()
    app.router.add_get("/",                  _handle_root)
    app.router.add_get("/paper_trades.json", _handle_trades)
    app.router.add_get("/health",            _handle_health)
    runner = aio_web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv("PORT", 8080))
    await aio_web.TCPSite(runner, "0.0.0.0", port).start()
    log_event(f"Dashboard running on port {port}")

# ==============================================================================
#  SELF-PING — keeps Render web service awake (pings /health every 10 min)
# ==============================================================================
async def self_ping():
    await asyncio.sleep(90)  # wait for server + first scan to complete
    port = int(os.getenv("PORT", 8080))
    while True:
        try:
            http = await get_http()
            await http.get(f"http://localhost:{port}/health")
            log_event("[PING] Keep-alive ping sent")
        except Exception as e:
            log_event(f"[PING ERROR] {e}")
        await asyncio.sleep(600)  # every 10 minutes

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
#  MEMORY PRUNING — called after every scan cycle
# ==============================================================================
def prune_memory():
    if len(state.log) > 80:
        state.log = state.log[-80:]

    open_pos   = [p for p in state.positions if p.status == "OPEN"]
    closed_pos = [p for p in state.positions if p.status == "CLOSED"][-10:]
    state.positions = open_pos + closed_pos

    if len(state.closed_trades) > 30:
        state.closed_trades = state.closed_trades[-30:]

    now = datetime.now(timezone.utc).timestamp()
    stale_keys = [k for k, t in _news_cache_time.items()
                  if now - t > NEWS_CACHE_TTL * 2]
    for k in stale_keys:
        _news_cache.pop(k, None)
        _news_cache_time.pop(k, None)

# ==============================================================================
#  POLYMARKET GAMMA API
# ==============================================================================
async def fetch_markets(limit: int = 50) -> list[Market]:
    try:
        http = await get_http()
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
                op  = m.get("outcomePrices")
                out = m.get("outcomes", '["Yes","No"]')
                if op:
                    prices   = json.loads(op)  if isinstance(op,  str) else op
                    outcomes = json.loads(out) if isinstance(out, str) else out
                    yi = next((i for i, o in enumerate(outcomes) if o.lower() in ("yes", "true")),  0)
                    ni = next((i for i, o in enumerate(outcomes) if o.lower() in ("no", "false")),  1)
                    yes_p, no_p = float(prices[yi]), float(prices[ni])
                else:
                    toks  = m.get("tokens", [])
                    y_tok = next((t for t in toks if str(t.get("outcome", "")).upper() == "YES"), None)
                    n_tok = next((t for t in toks if str(t.get("outcome", "")).upper() == "NO"),  None)
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
    base = [
        ("Will BTC close above $90k on Apr 15?",     0.42, 0.58, 2_100_000, "crypto"),
        ("Will ETH hit $4k before May 1?",            0.31, 0.69,   890_000, "crypto"),
        ("NBA: Lakers to win next game?",             0.55, 0.45,   340_000, "sports"),
        ("Will Fed cut rates in May 2026?",           0.38, 0.62, 1_200_000, "finance"),
        ("EPL: Arsenal to win vs Chelsea?",           0.48, 0.52,   230_000, "sports"),
        ("Will GPT-5 be announced before June?",     0.44, 0.56,   410_000, "tech"),
        ("NBA Finals: Will Celtics win 2026?",        0.28, 0.72, 1_900_000, "sports"),
        ("Will NVDA stock be above $900 by May?",    0.61, 0.39,   560_000, "finance"),
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
#  REPLICATE / LLAMA 3 70B
# ==============================================================================
async def call_replicate_llama(prompt: str) -> str:
    system = (
        "You are a quantitative prediction market analyst. "
        "Always respond with valid JSON only. No markdown fences, no extra text outside the JSON."
    )
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

    # Separate client with longer timeout — avoids polluting shared pool
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as http:
        r = await http.post(url, headers=headers, json=payload)
        r.raise_for_status()
        result = r.json()

        output = result.get("output")
        if isinstance(output, list):
            return "".join(output).strip()
        if isinstance(output, str):
            return output.strip()

        # Fall back to polling if "Prefer: wait" wasn't honoured
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
                raise ValueError(f"Replicate failed: {data.get('error')}")

        raise TimeoutError("Replicate timed out after 30s")


async def analyse_market(market: Market) -> Optional[dict]:
    news = await get_news_for_market(market.question, market.category)
    log_event(f"[NEWS] {market.question[:40]} -> {news[:80]}...")

    prompt = (
        f"Analyse this Polymarket prediction market contract.\n\n"
        f"Question : {market.question}\n"
        f"Category : {market.category}\n"
        f"YES price: {market.yes_price:.3f}  (market implies {market.yes_price*100:.1f}% probability)\n"
        f"NO  price: {market.no_price:.3f}\n"
        f"Volume 24h: ${market.volume:,.0f}\n"
        f"Description: {market.description or 'N/A'}\n\n"
        f"RECENT NEWS CONTEXT (use this to improve your estimate):\n"
        f"{news}\n\n"
        f"Based on your knowledge AND the news context above, estimate the TRUE probability of YES.\n\n"
        f'Respond ONLY with this exact JSON (no markdown, no extra text):\n'
        f'{{"true_prob_yes": 0.55, "confidence": 0.70, "reasoning": "one sentence", "data_quality": "HIGH"}}\n\n'
        f"data_quality: HIGH = well-known event with clear news, MEDIUM = some uncertainty, LOW = obscure/no news"
    )

    try:
        text = await call_replicate_llama(prompt)

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

        if dq == "HIGH":
            conf = min(conf + 0.05, 0.99)

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
#  SPREAD ARBITRAGE
# ==============================================================================
def detect_mispricing(market: Market) -> Optional[dict]:
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
    if confidence < 0.48:
        return False, f"Confidence {confidence:.2f} too low"
    if market.volume < 30_000:
        return False, f"Volume ${market.volume:,.0f} below $30k minimum"
    if market.yes_price < 0.05 or market.yes_price > 0.95:
        return False, f"Price {market.yes_price:.3f} too extreme -- skipping"
    if market.id in {p.market_id for p in open_positions}:
        return False, "Already have open position in this market"

    return True, "OK"


def position_size(edge: float, confidence: float) -> float:
    kelly = edge * confidence
    return min(state.balance * kelly,
               state.balance * MAX_POSITION_PCT,
               state.balance * 0.05)

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
    drift = random.gauss(0.002, 0.015)
    return max(0.01, min(0.99, pos.current_price + drift))


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
#  STATE PERSISTENCE — pushed to thread pool to avoid blocking event loop
# ==============================================================================
def _write_state():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "balance":          state.balance,
            "total_pnl":        state.total_pnl,
            "scan_count":       state.scan_count,
            "signals_analyzed": state.signals_analyzed,
            "trades_taken":     state.trades_taken,
            "start_time":       state.start_time,
            "open_positions":   [asdict(p) for p in state.positions if p.status == "OPEN"],
            "closed_trades":    state.closed_trades[-50:],
            "log":              state.log[-100:],
            "updated_at":       datetime.now(timezone.utc).isoformat(),
        }, f, indent=2, ensure_ascii=False)


async def save_state():
    await asyncio.to_thread(_write_state)


def log_event(msg: str):
    ts    = datetime.now(timezone.utc).strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    state.log.append(entry)
    if len(state.log) > 80:
        state.log = state.log[-80:]
    print(entry)

# ==============================================================================
#  MAIN SCAN LOOP
# ==============================================================================
async def scan_cycle():
    state.scan_count += 1
    log_event(f"-- Scan #{state.scan_count} " + "-" * 35)

    markets = await fetch_markets(50)
    update_positions(markets)

    open_ids = {p.market_id for p in state.positions if p.status == "OPEN"}

    def question_allowed(q: str) -> bool:
        t = q.lower()
        return any(kw in t for kw in [
            "win", "nba", "nfl", "fifa", "epl", "soccer", "football",
            "basketball", "tennis", "golf", "cricket", "ufc", "boxing",
            "baseball", "hockey", "mls", "championship", "league", "cup",
            "match", "game", "season", "playoff", "finals", "tournament",
            "bitcoin", "btc", "ethereum", "eth", "crypto", "solana",
            "coinbase", "binance", "blockchain", "defi", "nft",
            "president", "election", "vote", "prime minister", "senate",
            "congress", "trump", "policy", "war", "ceasefire", "iran",
            "ukraine", "china", "russia", "tariff",
            "fed", "rate", "inflation", "gdp", "stock", "market",
            "nasdaq", "s&p", "recession", "oil", "gold",
        ])

    candidates = sorted(
        [
            m for m in markets
            if m.id not in open_ids
            and m.volume > 30_000
            and question_allowed(m.question)
        ],
        key=lambda m: m.volume, reverse=True
    )[:5]

    log_event(f"[SCAN] {len(candidates)} candidates")

    for market in candidates:
        state.signals_analyzed += 1

        arb    = detect_mispricing(market)
        signal = await analyse_market(market)

        if not signal:
            continue

        if arb:
            signal["edge"]       = max(signal["edge"], arb["gap"] * 0.5)
            signal["confidence"] = min(signal["confidence"] + 0.1, 0.95)
            log_event(f"[ARB] Spread anomaly on {market.question[:40]}")

        allow, reason = risk_check(signal["edge"], signal["confidence"], market)
        if not allow:
            log_event(f"[SKIP] {market.question[:45]} -- {reason}")
            continue

        open_position(market, signal)
        await asyncio.sleep(1)

    prune_memory()
    await save_state()
    log_event(
        f"Balance: ${state.balance:.2f} | "
        f"PnL: ${state.total_pnl:+.2f} | "
        f"Open: {sum(1 for p in state.positions if p.status == 'OPEN')}"
    )

# ==============================================================================
#  BOT LOOP
# ==============================================================================
async def bot_loop():
    await start_web()
    asyncio.create_task(self_ping())  # keeps Render web service awake

    log_event("Polymarket AI Paper Trading Bot started")
    log_event(f"LLM    : Replicate -> {REPLICATE_MODEL}")
    log_event(f"Balance: ${state.balance:.2f} USDC (PAPER)")
    log_event(f"Scan interval: {SCAN_INTERVAL}s | Max positions: {MAX_OPEN_POSITIONS}")
    log_event(f"Candidates per scan: 5 | RSS feeds per market: 2")
    log_event("-" * 60)

    while True:
        try:
            await scan_cycle()
        except Exception as e:
            log_event(f"[LOOP ERROR] {e}")
        await asyncio.sleep(SCAN_INTERVAL)

# ==============================================================================
#  ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    asyncio.run(bot_loop())