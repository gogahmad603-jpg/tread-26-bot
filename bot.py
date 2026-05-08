import os
import asyncio
import logging
import math
import requests
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "")

PAIRS = [
    {"symbol": "EUR/USD", "label": "EUR → USD 🇪🇺🇺🇸", "hot": True,  "otc": True},
    {"symbol": "GBP/USD", "label": "GBP → USD 🇬🇧🇺🇸", "hot": True,  "otc": True},
    {"symbol": "USD/JPY", "label": "USD → JPY 🇺🇸🇯🇵", "hot": True,  "otc": True},
    {"symbol": "AUD/USD", "label": "AUD → USD 🇦🇺🇺🇸", "hot": False, "otc": True},
    {"symbol": "USD/CHF", "label": "USD → CHF 🇺🇸🇨🇭", "hot": False, "otc": True},
    {"symbol": "EUR/GBP", "label": "EUR → GBP 🇪🇺🇬🇧", "hot": True,  "otc": False},
    {"symbol": "USD/CAD", "label": "USD → CAD 🇺🇸🇨🇦", "hot": False, "otc": True},
    {"symbol": "NZD/USD", "label": "NZD → USD 🇳🇿🇺🇸", "hot": False, "otc": False},
]

TIMEFRAMES = [
    {"key": "1min",  "label": "1m",  "desc": "1 Minute",   "emoji": "🏃"},
    {"key": "5min",  "label": "5m",  "desc": "5 Minutes",  "emoji": "🏅"},
    {"key": "15min", "label": "15m", "desc": "15 Minutes", "emoji": "⭐"},
    {"key": "30min", "label": "30m", "desc": "30 Minutes", "emoji": "🎯"},
    {"key": "1h",    "label": "1h",  "desc": "1 Hour",     "emoji": "🏆"},
]

# =================== PURE PYTHON MATH ===================

def mean(data):
    return sum(data) / len(data) if data else 0

def stdev(data):
    if len(data) < 2:
        return 0
    m = mean(data)
    return math.sqrt(sum((x - m) ** 2 for x in data) / len(data))

def ema(data, period):
    if len(data) < period:
        return []
    k = 2 / (period + 1)
    result = [mean(data[:period])]
    for price in data[period:]:
        result.append(price * k + result[-1] * (1 - k))
    return result

def rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i-1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = mean(gains[-period:])
    avg_loss = mean(losses[-period:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal:
        return 0, 0, 0
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    min_len = min(len(ema_fast), len(ema_slow))
    macd_line = [ema_fast[-(min_len-i)] - ema_slow[-(min_len-i)] for i in range(min_len)]
    if len(macd_line) < signal:
        return 0, 0, 0
    signal_line = ema(macd_line, signal)
    if not signal_line:
        return 0, 0, 0
    ml = macd_line[-1]
    sl = signal_line[-1]
    hist = ml - sl
    return ml, sl, hist

def bollinger(closes, period=20, mult=2.0):
    if len(closes) < period:
        return 0, 0, 0
    recent = closes[-period:]
    mid = mean(recent)
    sd = stdev(recent)
    return mid + mult * sd, mid, mid - mult * sd

def stochastic(highs, lows, closes, k=14, d=3):
    if len(closes) < k:
        return 50, 50
    h = max(highs[-k:])
    l = min(lows[-k:])
    if h == l:
        return 50, 50
    stoch_k = 100 * (closes[-1] - l) / (h - l)
    stoch_d = mean([100 * (closes[-i] - min(lows[-k-i:-i] if len(lows) >= k+i else lows)) /
                    max(1, max(highs[-k-i:-i] if len(highs) >= k+i else highs) -
                        min(lows[-k-i:-i] if len(lows) >= k+i else lows))
                    for i in range(1, d+1)])
    return stoch_k, stoch_d

def zigzag_last(closes, threshold=0.003):
    if len(closes) < 10:
        return None
    direction = None
    last_pivot = closes[0]
    last_type = None
    last_idx = 0
    for i in range(1, len(closes)):
        change = (closes[i] - last_pivot) / last_pivot if last_pivot != 0 else 0
        if direction is None:
            if abs(change) >= threshold:
                direction = "up" if change > 0 else "down"
                last_pivot = closes[i]
                last_idx = i
        elif direction == "up":
            if change <= -threshold:
                last_type = "high"
                direction = "down"
                last_pivot = closes[i]
                last_idx = i
            elif closes[i] > last_pivot:
                last_pivot = closes[i]
                last_idx = i
        elif direction == "down":
            if change >= threshold:
                last_type = "low"
                direction = "up"
                last_pivot = closes[i]
                last_idx = i
            elif closes[i] < last_pivot:
                last_pivot = closes[i]
                last_idx = i
    if last_type:
        return last_type, last_idx
    return None

def volume_trend(volumes, period=10):
    if not volumes or sum(volumes) == 0:
        return "normal"
    avg = mean(volumes[-period:]) if len(volumes) >= period else mean(volumes)
    last = volumes[-1]
    if avg == 0:
        return "normal"
    if last > avg * 1.3:
        return "high"
    if last < avg * 0.7:
        return "low"
    return "normal"

# =================== DATA FETCH ===================

def fetch_data(symbol, interval, size=100):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol, "interval": interval,
        "outputsize": size, "apikey": TWELVE_DATA_KEY, "format": "JSON"
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if "values" not in data:
            return None
        values = list(reversed(data["values"]))
        closes  = [float(v["close"])  for v in values]
        highs   = [float(v["high"])   for v in values]
        lows    = [float(v["low"])    for v in values]
        volumes = [float(v.get("volume", 0)) for v in values]
        return closes, highs, lows, volumes
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return None

# =================== ANALYSIS ===================

def analyze(symbol, interval):
    result = fetch_data(symbol, interval)
    if not result or len(result[0]) < 40:
        return {"error": "لا توجد بيانات كافية"}

    closes, highs, lows, volumes = result
    price = closes[-1]

    rsi_val   = rsi(closes)
    ml, sl, mh = macd(closes)
    bb_up, bb_mid, bb_low = bollinger(closes)
    e9  = ema(closes, 9)[-1]  if len(ema(closes, 9))  > 0 else price
    e21 = ema(closes, 21)[-1] if len(ema(closes, 21)) > 0 else price
    e50 = ema(closes, 50)[-1] if len(ema(closes, 50)) > 0 else price
    stk, std_val = stochastic(highs, lows, closes)
    vol = volume_trend(volumes)
    zz  = zigzag_last(closes)

    score = 0
    max_score = 0
    signals = []

    # RSI (20)
    max_score += 20
    if rsi_val < 30:
        score += 20; signals.append(("✅", "RSI ذروة بيع قوية", "UP"))
    elif rsi_val < 40:
        score += 12; signals.append(("🟡", "RSI منطقة بيع", "UP"))
    elif rsi_val > 70:
        score -= 20; signals.append(("✅", "RSI ذروة شراء قوية", "DOWN"))
    elif rsi_val > 60:
        score -= 12; signals.append(("🟡", "RSI منطقة شراء", "DOWN"))
    else:
        signals.append(("⚪", "RSI محايد", "NEUTRAL"))

    # MACD (20)
    max_score += 20
    if ml > sl and mh > 0:
        score += 20; signals.append(("✅", "MACD تقاطع صاعد", "UP"))
    elif ml > sl:
        score += 10; signals.append(("🟡", "MACD فوق خط الإشارة", "UP"))
    elif ml < sl and mh < 0:
        score -= 20; signals.append(("✅", "MACD تقاطع هابط", "DOWN"))
    elif ml < sl:
        score -= 10; signals.append(("🟡", "MACD تحت خط الإشارة", "DOWN"))

    # Bollinger (15)
    max_score += 15
    bb_range = bb_up - bb_low
    bb_pos = (price - bb_low) / bb_range if bb_range != 0 else 0.5
    if bb_pos < 0.15:
        score += 15; signals.append(("✅", "السعر عند البولينجر السفلي", "UP"))
    elif bb_pos < 0.3:
        score += 8;  signals.append(("🟡", "قريب من البولينجر السفلي", "UP"))
    elif bb_pos > 0.85:
        score -= 15; signals.append(("✅", "السعر عند البولينجر العلوي", "DOWN"))
    elif bb_pos > 0.7:
        score -= 8;  signals.append(("🟡", "قريب من البولينجر العلوي", "DOWN"))
    else:
        signals.append(("⚪", "السعر في منتصف البولينجر", "NEUTRAL"))

    # EMA (20)
    max_score += 20
    if price > e9 > e21 > e50:
        score += 20; signals.append(("✅", "EMA تراص صاعد كامل", "UP"))
    elif price > e9 and e9 > e21:
        score += 12; signals.append(("🟡", "EMA اتجاه صاعد", "UP"))
    elif price < e9 < e21 < e50:
        score -= 20; signals.append(("✅", "EMA تراص هابط كامل", "DOWN"))
    elif price < e9 and e9 < e21:
        score -= 12; signals.append(("🟡", "EMA اتجاه هابط", "DOWN"))
    else:
        signals.append(("⚪", "EMA متقاطع محايد", "NEUTRAL"))

    # Stochastic (15)
    max_score += 15
    if stk < 20 and std_val < 20:
        score += 15; signals.append(("✅", "Stochastic ذروة بيع", "UP"))
    elif stk > 80 and std_val > 80:
        score -= 15; signals.append(("✅", "Stochastic ذروة شراء", "DOWN"))
    elif stk < 30:
        score += 8;  signals.append(("🟡", "Stochastic منطقة شراء", "UP"))
    elif stk > 70:
        score -= 8;  signals.append(("🟡", "Stochastic منطقة بيع", "DOWN"))
    else:
        signals.append(("⚪", "Stochastic محايد", "NEUTRAL"))

    # Volume (10)
    max_score += 10
    if vol == "high":
        if score > 0:
            score += 10; signals.append(("✅", "حجم مرتفع يدعم الصعود", "UP"))
        else:
            score -= 10; signals.append(("✅", "حجم مرتفع يدعم الهبوط", "DOWN"))
    else:
        signals.append(("⚪", f"حجم تداول {vol}", "NEUTRAL"))

    # ZigZag (10)
    max_score += 10
    if zz:
        zz_type, zz_idx = zz
        recent = len(closes) - zz_idx
        if zz_type == "low" and recent <= 5:
            score += 10; signals.append(("✅", "ZigZag: قاع جديد → ارتداد", "UP"))
        elif zz_type == "high" and recent <= 5:
            score -= 10; signals.append(("✅", "ZigZag: قمة جديدة → تراجع", "DOWN"))
        else:
            signals.append(("⚪", "ZigZag: لا نقطة محورية حديثة", "NEUTRAL"))
    else:
        signals.append(("⚪", "ZigZag: بيانات غير كافية", "NEUTRAL"))

    confidence = abs(score) / max_score * 100
    direction = "UP" if score > 0 else "DOWN" if score < 0 else "NEUTRAL"
    strength = "ضعيفة 🔴" if confidence < 40 else \
               "متوسطة 🟡" if confidence < 60 else \
               "قوية 🟢"   if confidence < 80 else "قوية جداً ⚡"

    return {
        "direction": direction, "confidence": round(confidence, 1),
        "strength": strength, "signals": signals,
        "rsi": round(rsi_val, 1), "price": round(price, 5),
        "stoch_k": round(stk, 1), "vol_trend": vol,
        "e9": round(e9, 5), "e21": round(e21, 5),
    }

# =================== MESSAGE ===================

def build_msg(symbol, interval, r):
    d = r["direction"]
    arrow = "⬆️" if d == "UP" else "⬇️" if d == "DOWN" else "↔️"
    color = "🟢" if d == "UP" else "🔴" if d == "DOWN" else "⚪"
    conf = r["confidence"]
    bar = "█" * int(conf/10) + "░" * (10 - int(conf/10))
    sigs = "".join(f"  {e} {desc}\n" for e, desc, side in r["signals"] if side != "NEUTRAL")

    return f"""
╔══════════════════════════════╗
║   📊 *TRADE ALGO BOT* 📊      ║
╚══════════════════════════════╝

🔷 *الزوج:* `{symbol}`
⏱ *الفريم:* `{interval}`
💰 *السعر:* `{r['price']}`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{color} *الإشارة: {arrow} {d}*

📈 *قوة الإشارة:* {r['strength']}
🎯 *نسبة الثقة:* `{conf}%`
`[{bar}]`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 *المؤشرات النشطة:*
{sigs}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 *تفاصيل:*
• RSI: `{r['rsi']}` | Stoch: `{r['stoch_k']}`
• Volume: `{r['vol_trend'].upper()}`
• EMA9: `{r['e9']}` | EMA21: `{r['e21']}`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🕐 {datetime.now().strftime('%H:%M:%S')} | Trade Algo Bot ⚡
""".strip()

# =================== KEYBOARDS ===================

def pairs_kb():
    rows = []
    for i in range(0, len(PAIRS), 2):
        row = []
        for p in PAIRS[i:i+2]:
            tags = ("🔥" if p["hot"] else "") + (" OTC" if p["otc"] else "")
            row.append(InlineKeyboardButton(f"{p['label']} {tags}", callback_data=f"pair:{p['symbol']}"))
        rows.append(row)
    rows.append([InlineKeyboardButton("🔙 رجوع", callback_data="start")])
    return InlineKeyboardMarkup(rows)

def tf_kb(symbol):
    rows = [[InlineKeyboardButton(f"{tf['emoji']} {tf['label']} — {tf['desc']}",
             callback_data=f"tf:{symbol}:{tf['key']}")] for tf in TIMEFRAMES]
    rows.append([InlineKeyboardButton("🔙 رجوع للأزواج", callback_data="choose_pair")])
    return InlineKeyboardMarkup(rows)

def signal_kb(symbol, interval):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 تحليل جديد", callback_data=f"tf:{symbol}:{interval}")],
        [InlineKeyboardButton("🔀 تغيير الزوج", callback_data="choose_pair")],
        [InlineKeyboardButton("🏠 القائمة الرئيسية", callback_data="start")],
    ])

# =================== HANDLERS ===================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("📊 اختر زوج العملات", callback_data="choose_pair")]])
    text = """
╔══════════════════════════════╗
║   📈 *TRADE ALGO BOT* 📈      ║
║   *Advanced Trading Signals*  ║
╚══════════════════════════════╝

مرحباً! 🚀 بوت الإشارات المتقدم

*المؤشرات:*
✅ RSI ✅ MACD ✅ Bollinger
✅ EMA 9/21/50 ✅ Stochastic
✅ Volume ✅ ZigZag

اضغط للبدء ⬇️
""".strip()
    if update.message:
        await update.message.reply_text(text, reply_markup=kb, parse_mode=ParseMode.MARKDOWN)
    elif update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=kb, parse_mode=ParseMode.MARKDOWN)

async def cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    d = q.data

    if d == "start":
        await start(update, context)
    elif d == "choose_pair":
        await q.edit_message_text("💱 *اختر زوج العملات:*\n🔥 = الأكثر تداولاً | OTC = خارج السوق",
                                   reply_markup=pairs_kb(), parse_mode=ParseMode.MARKDOWN)
    elif d.startswith("pair:"):
        symbol = d.split(":")[1]
        await q.edit_message_text(f"⏱ *اختر الإطار الزمني:*\n📌 الزوج: `{symbol}`\n💡 الموصى به: 1 دقيقة",
                                   reply_markup=tf_kb(symbol), parse_mode=ParseMode.MARKDOWN)
    elif d.startswith("tf:"):
        _, symbol, interval = d.split(":")
        await q.edit_message_text(f"⏳ *جاري التحليل...*\n🔍 `{symbol}` | `{interval}`\nانتظر... ⚙️",
                                   parse_mode=ParseMode.MARKDOWN)
        result = analyze(symbol, interval)
        if "error" in result:
            await q.edit_message_text(f"❌ {result['error']}", reply_markup=signal_kb(symbol, interval))
        else:
            await q.edit_message_text(build_msg(symbol, interval, result),
                                       reply_markup=signal_kb(symbol, interval),
                                       parse_mode=ParseMode.MARKDOWN)

# =================== MAIN ===================

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(cb))
    logger.info("🚀 Bot running...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
