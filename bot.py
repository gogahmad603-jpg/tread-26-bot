import os
import asyncio
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode
import requests
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== CONFIG =====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_HERE")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "YOUR_TWELVE_DATA_KEY_HERE")

# ===================== PAIRS =====================
PAIRS = [
    {"symbol": "EUR/USD", "label": "EUR → USD 🇪🇺🇺🇸", "hot": True, "otc": True},
    {"symbol": "GBP/USD", "label": "GBP → USD 🇬🇧🇺🇸", "hot": True, "otc": True},
    {"symbol": "USD/JPY", "label": "USD → JPY 🇺🇸🇯🇵", "hot": True, "otc": True},
    {"symbol": "AUD/USD", "label": "AUD → USD 🇦🇺🇺🇸", "hot": False, "otc": True},
    {"symbol": "USD/CHF", "label": "USD → CHF 🇺🇸🇨🇭", "hot": False, "otc": True},
    {"symbol": "EUR/GBP", "label": "EUR → GBP 🇪🇺🇬🇧", "hot": True, "otc": False},
    {"symbol": "USD/CAD", "label": "USD → CAD 🇺🇸🇨🇦", "hot": False, "otc": True},
    {"symbol": "NZD/USD", "label": "NZD → USD 🇳🇿🇺🇸", "hot": False, "otc": False},
]

# ===================== TIMEFRAMES =====================
TIMEFRAMES = [
    {"key": "1min",  "label": "1m",  "desc": "1 Minute",   "emoji": "🏃"},
    {"key": "5min",  "label": "5m",  "desc": "5 Minutes",  "emoji": "🏅"},
    {"key": "15min", "label": "15m", "desc": "15 Minutes", "emoji": "⭐"},
    {"key": "30min", "label": "30m", "desc": "30 Minutes", "emoji": "🎯"},
    {"key": "1h",    "label": "1h",  "desc": "1 Hour",     "emoji": "🏆"},
]

USER_STATE = {}

# ===================== DATA FETCHING =====================
def fetch_ohlcv(symbol: str, interval: str, outputsize: int = 100):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_KEY,
        "format": "JSON"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if "values" not in data:
            return None
        df = pd.DataFrame(data["values"])
        df = df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"})
        for col in ["open","high","low","close"]:
            df[col] = pd.to_numeric(df[col])
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Data fetch error: {e}")
        return None

# ===================== INDICATORS =====================
def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calc_bollinger(close: pd.Series, period: int = 20, std: float = 2.0):
    sma = close.rolling(period).mean()
    stddev = close.rolling(period).std()
    upper = sma + std * stddev
    lower = sma - std * stddev
    return upper, sma, lower

def calc_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()

def calc_stoch(high, low, close, k=14, d=3):
    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()
    stoch_k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d

def calc_zigzag(close: pd.Series, threshold: float = 0.003):
    pivots = []
    direction = None
    last_pivot = close.iloc[0]
    last_idx = 0
    for i in range(1, len(close)):
        change = (close.iloc[i] - last_pivot) / last_pivot
        if direction is None:
            if abs(change) >= threshold:
                direction = "up" if change > 0 else "down"
                last_pivot = close.iloc[i]
                last_idx = i
        elif direction == "up":
            if change <= -threshold:
                pivots.append(("high", last_idx, last_pivot))
                direction = "down"
                last_pivot = close.iloc[i]
                last_idx = i
            elif close.iloc[i] > last_pivot:
                last_pivot = close.iloc[i]
                last_idx = i
        elif direction == "down":
            if change >= threshold:
                pivots.append(("low", last_idx, last_pivot))
                direction = "up"
                last_pivot = close.iloc[i]
                last_idx = i
            elif close.iloc[i] < last_pivot:
                last_pivot = close.iloc[i]
                last_idx = i
    return pivots

def calc_volume_trend(volume: pd.Series, period: int = 10):
    if volume.sum() == 0:
        return "neutral"
    avg_vol = volume.rolling(period).mean().iloc[-1]
    last_vol = volume.iloc[-1]
    if last_vol > avg_vol * 1.3:
        return "high"
    elif last_vol < avg_vol * 0.7:
        return "low"
    return "normal"

# ===================== MAIN ANALYSIS ENGINE =====================
def analyze(symbol: str, interval: str) -> dict:
    df = fetch_ohlcv(symbol, interval, outputsize=100)
    if df is None or len(df) < 40:
        return {"error": "لا توجد بيانات كافية"}

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    volume= df["volume"] if "volume" in df.columns else pd.Series([0]*len(df))

    # --- Indicators ---
    rsi = calc_rsi(close, 14).iloc[-1]
    macd_line, macd_sig, macd_hist = calc_macd(close)
    ml, ms, mh = macd_line.iloc[-1], macd_sig.iloc[-1], macd_hist.iloc[-1]
    bb_up, bb_mid, bb_low = calc_bollinger(close)
    bb_up, bb_low = bb_up.iloc[-1], bb_low.iloc[-1]
    price = close.iloc[-1]
    ema9  = calc_ema(close, 9).iloc[-1]
    ema21 = calc_ema(close, 21).iloc[-1]
    ema50 = calc_ema(close, 50).iloc[-1]
    stk, std = calc_stoch(high, low, close)
    stk, std = stk.iloc[-1], std.iloc[-1]
    vol_trend = calc_volume_trend(volume)
    zigzags = calc_zigzag(close)
    zz_last = zigzags[-1] if zigzags else None

    # ===================== SCORING SYSTEM =====================
    score = 0      # positive = UP, negative = DOWN
    signals = []   # list of signal descriptions
    max_score = 0

    # 1. RSI (weight: 20)
    max_score += 20
    if rsi < 30:
        score += 20
        signals.append(("✅", "RSI ذروة بيع قوية", "UP"))
    elif rsi < 40:
        score += 12
        signals.append(("🟡", "RSI منطقة بيع", "UP"))
    elif rsi > 70:
        score -= 20
        signals.append(("✅", "RSI ذروة شراء قوية", "DOWN"))
    elif rsi > 60:
        score -= 12
        signals.append(("🟡", "RSI منطقة شراء", "DOWN"))
    else:
        signals.append(("⚪", "RSI محايد", "NEUTRAL"))

    # 2. MACD (weight: 20)
    max_score += 20
    if ml > ms and mh > 0:
        score += 20
        signals.append(("✅", "MACD تقاطع صاعد", "UP"))
    elif ml > ms:
        score += 10
        signals.append(("🟡", "MACD فوق خط الإشارة", "UP"))
    elif ml < ms and mh < 0:
        score -= 20
        signals.append(("✅", "MACD تقاطع هابط", "DOWN"))
    elif ml < ms:
        score -= 10
        signals.append(("🟡", "MACD تحت خط الإشارة", "DOWN"))

    # 3. Bollinger Bands (weight: 15)
    max_score += 15
    bb_range = bb_up - bb_low
    bb_pos = (price - bb_low) / bb_range if bb_range != 0 else 0.5
    if bb_pos < 0.15:
        score += 15
        signals.append(("✅", "السعر عند البولينجر السفلي", "UP"))
    elif bb_pos < 0.3:
        score += 8
        signals.append(("🟡", "السعر قريب من البولينجر السفلي", "UP"))
    elif bb_pos > 0.85:
        score -= 15
        signals.append(("✅", "السعر عند البولينجر العلوي", "DOWN"))
    elif bb_pos > 0.7:
        score -= 8
        signals.append(("🟡", "السعر قريب من البولينجر العلوي", "DOWN"))
    else:
        signals.append(("⚪", "السعر في منتصف البولينجر", "NEUTRAL"))

    # 4. EMA Alignment (weight: 20)
    max_score += 20
    if price > ema9 > ema21 > ema50:
        score += 20
        signals.append(("✅", "EMA تراص صاعد كامل", "UP"))
    elif price > ema9 and ema9 > ema21:
        score += 12
        signals.append(("🟡", "EMA اتجاه صاعد", "UP"))
    elif price < ema9 < ema21 < ema50:
        score -= 20
        signals.append(("✅", "EMA تراص هابط كامل", "DOWN"))
    elif price < ema9 and ema9 < ema21:
        score -= 12
        signals.append(("🟡", "EMA اتجاه هابط", "DOWN"))
    else:
        signals.append(("⚪", "EMA متقاطع محايد", "NEUTRAL"))

    # 5. Stochastic (weight: 15)
    max_score += 15
    if stk < 20 and std < 20:
        score += 15
        signals.append(("✅", "Stochastic ذروة بيع", "UP"))
    elif stk > 80 and std > 80:
        score -= 15
        signals.append(("✅", "Stochastic ذروة شراء", "DOWN"))
    elif stk < 30:
        score += 8
        signals.append(("🟡", "Stochastic منطقة شراء", "UP"))
    elif stk > 70:
        score -= 8
        signals.append(("🟡", "Stochastic منطقة بيع", "DOWN"))
    else:
        signals.append(("⚪", "Stochastic محايد", "NEUTRAL"))

    # 6. Volume (weight: 10)
    max_score += 10
    if vol_trend == "high":
        if score > 0:
            score += 10
            signals.append(("✅", "حجم تداول مرتفع يدعم الصعود", "UP"))
        else:
            score -= 10
            signals.append(("✅", "حجم تداول مرتفع يدعم الهبوط", "DOWN"))
    elif vol_trend == "low":
        signals.append(("⚪", "حجم تداول منخفض (إشارة ضعيفة)", "NEUTRAL"))
    else:
        signals.append(("⚪", "حجم تداول طبيعي", "NEUTRAL"))

    # 7. ZigZag (bonus weight: 10)
    max_score += 10
    if zz_last:
        zz_type, zz_idx, zz_val = zz_last
        recent = len(close) - zz_idx
        if zz_type == "low" and recent <= 5:
            score += 10
            signals.append(("✅", "ZigZag: قاع محلي جديد → ارتداد محتمل", "UP"))
        elif zz_type == "high" and recent <= 5:
            score -= 10
            signals.append(("✅", "ZigZag: قمة محلية جديدة → تراجع محتمل", "DOWN"))
        else:
            signals.append(("⚪", "ZigZag: لا نقطة محورية حديثة", "NEUTRAL"))

    # ===================== FINAL DECISION =====================
    confidence = abs(score) / max_score * 100
    direction = "UP" if score > 0 else "DOWN" if score < 0 else "NEUTRAL"

    strength = "ضعيفة 🔴" if confidence < 40 else \
               "متوسطة 🟡" if confidence < 60 else \
               "قوية 🟢" if confidence < 80 else \
               "قوية جداً ⚡"

    return {
        "direction": direction,
        "confidence": round(confidence, 1),
        "strength": strength,
        "score": score,
        "signals": signals,
        "rsi": round(rsi, 1),
        "price": round(price, 5),
        "macd_hist": round(mh, 6),
        "stoch_k": round(stk, 1),
        "vol_trend": vol_trend,
        "ema9": round(ema9, 5),
        "ema21": round(ema21, 5),
    }

# ===================== MESSAGE BUILDERS =====================
def build_signal_message(symbol: str, interval: str, result: dict) -> str:
    d = result["direction"]
    arrow = "⬆️" if d == "UP" else "⬇️" if d == "DOWN" else "↔️"
    color = "🟢" if d == "UP" else "🔴" if d == "DOWN" else "⚪"
    conf = result["confidence"]

    conf_bar = ""
    filled = int(conf / 10)
    conf_bar = "█" * filled + "░" * (10 - filled)

    signals_text = ""
    for emoji, desc, side in result["signals"]:
        if side != "NEUTRAL":
            signals_text += f"  {emoji} {desc}\n"

    msg = f"""
╔══════════════════════════════╗
║   📊 *TRADE ALGO BOT* 📊      ║
╚══════════════════════════════╝

🔷 *الزوج:* `{symbol}`
⏱ *الفريم:* `{interval}`
💰 *السعر:* `{result['price']}`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{color} *الإشارة: {arrow} {d}*

📈 *قوة الإشارة:* {result['strength']}
🎯 *نسبة الثقة:* `{conf}%`
`[{conf_bar}]`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 *المؤشرات النشطة:*
{signals_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 *تفاصيل سريعة:*
• RSI: `{result['rsi']}`
• Stochastic K: `{result['stoch_k']}`
• Volume: `{result['vol_trend'].upper()}`
• EMA9: `{result['ema9']}` | EMA21: `{result['ema21']}`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🕐 {datetime.now().strftime('%H:%M:%S')} | Trade Algo Bot ⚡
"""
    return msg.strip()

# ===================== KEYBOARDS =====================
def pairs_keyboard():
    rows = []
    for i in range(0, len(PAIRS), 2):
        row = []
        for p in PAIRS[i:i+2]:
            tags = ""
            if p["hot"]: tags += "🔥"
            if p["otc"]: tags += " OTC"
            row.append(InlineKeyboardButton(
                f"{p['label']} {tags}",
                callback_data=f"pair:{p['symbol']}"
            ))
        rows.append(row)
    rows.append([InlineKeyboardButton("🔙 رجوع", callback_data="start")])
    return InlineKeyboardMarkup(rows)

def timeframe_keyboard(symbol: str):
    rows = []
    for tf in TIMEFRAMES:
        rows.append([InlineKeyboardButton(
            f"{tf['emoji']} {tf['label']} — {tf['desc']}",
            callback_data=f"tf:{symbol}:{tf['key']}"
        )])
    rows.append([InlineKeyboardButton("🔙 رجوع للأزواج", callback_data="choose_pair")])
    return InlineKeyboardMarkup(rows)

def signal_keyboard(symbol: str, interval: str):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 تحليل جديد", callback_data=f"tf:{symbol}:{interval}")],
        [InlineKeyboardButton("🔀 تغيير الزوج", callback_data="choose_pair")],
        [InlineKeyboardButton("🏠 القائمة الرئيسية", callback_data="start")],
    ])

# ===================== HANDLERS =====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 اختر زوج العملات", callback_data="choose_pair")],
    ])
    text = """
╔══════════════════════════════╗
║   📈 *TRADE ALGO BOT* 📈      ║
║   *Advanced Trading Signals*  ║
╚══════════════════════════════╝

مرحباً بك في بوت الإشارات المتقدم! 🚀

*المؤشرات المستخدمة:*
✅ RSI — مؤشر القوة النسبية
✅ MACD — الزخم والاتجاه
✅ Bollinger Bands — تقلبات السعر
✅ EMA 9/21/50 — المتوسطات
✅ Stochastic — ذروة الشراء/البيع
✅ Volume — حجم التداول
✅ ZigZag — النقاط المحورية

اضغط للبدء ⬇️
"""
    if update.message:
        await update.message.reply_text(text, reply_markup=kb, parse_mode=ParseMode.MARKDOWN)
    elif update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=kb, parse_mode=ParseMode.MARKDOWN)

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "start":
        await start(update, context)

    elif data == "choose_pair":
        await query.edit_message_text(
            "💱 *اختر زوج العملات:*\n\n🔥 = الأكثر تداولاً | OTC = خارج السوق",
            reply_markup=pairs_keyboard(),
            parse_mode=ParseMode.MARKDOWN
        )

    elif data.startswith("pair:"):
        symbol = data.split(":")[1]
        await query.edit_message_text(
            f"⏱ *اختر الإطار الزمني:*\n\n📌 الزوج المختار: `{symbol}`\n\n💡 الموصى به: 1 دقيقة للإشارات السريعة",
            reply_markup=timeframe_keyboard(symbol),
            parse_mode=ParseMode.MARKDOWN
        )

    elif data.startswith("tf:"):
        parts = data.split(":")
        symbol = parts[1]
        interval = parts[2]
        await query.edit_message_text(
            f"⏳ *جاري التحليل...*\n\n🔍 تحليل `{symbol}` على فريم `{interval}`\n\nانتظر لحظة... ⚙️",
            parse_mode=ParseMode.MARKDOWN
        )
        result = analyze(symbol, interval)
        if "error" in result:
            await query.edit_message_text(
                f"❌ خطأ: {result['error']}\n\nحاول مرة أخرى.",
                reply_markup=signal_keyboard(symbol, interval)
            )
        else:
            msg = build_signal_message(symbol, interval, result)
            await query.edit_message_text(
                msg,
                reply_markup=signal_keyboard(symbol, interval),
                parse_mode=ParseMode.MARKDOWN
            )

# ===================== MAIN =====================
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(callback_handler))
    logger.info("🚀 Trade Algo Bot is running...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
