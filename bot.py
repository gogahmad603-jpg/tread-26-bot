import os
import math
import logging
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
    {"symbol": "EUR/USD", "label": "EUR/USD 🇪🇺🇺🇸", "hot": True,  "otc": True},
    {"symbol": "GBP/USD", "label": "GBP/USD 🇬🇧🇺🇸", "hot": True,  "otc": True},
    {"symbol": "USD/JPY", "label": "USD/JPY 🇺🇸🇯🇵", "hot": True,  "otc": True},
    {"symbol": "AUD/USD", "label": "AUD/USD 🇦🇺🇺🇸", "hot": True,  "otc": True},
    {"symbol": "USD/CHF", "label": "USD/CHF 🇺🇸🇨🇭", "hot": False, "otc": True},
    {"symbol": "EUR/GBP", "label": "EUR/GBP 🇪🇺🇬🇧", "hot": True,  "otc": False},
    {"symbol": "USD/CAD", "label": "USD/CAD 🇺🇸🇨🇦", "hot": False, "otc": True},
    {"symbol": "NZD/USD", "label": "NZD/USD 🇳🇿🇺🇸", "hot": False, "otc": False},
    {"symbol": "EUR/JPY", "label": "EUR/JPY 🇪🇺🇯🇵", "hot": True,  "otc": True},
    {"symbol": "GBP/JPY", "label": "GBP/JPY 🇬🇧🇯🇵", "hot": True,  "otc": True},
]

TIMEFRAMES = [
    {"key": "1min",  "label": "1m",  "desc": "1 دقيقة",    "emoji": "⚡", "rec": True},
    {"key": "5min",  "label": "5m",  "desc": "5 دقائق",    "emoji": "🏅", "rec": False},
    {"key": "15min", "label": "15m", "desc": "15 دقيقة",   "emoji": "⭐", "rec": False},
    {"key": "30min", "label": "30m", "desc": "30 دقيقة",   "emoji": "🎯", "rec": False},
    {"key": "1h",    "label": "1h",  "desc": "ساعة",        "emoji": "🏆", "rec": False},
]

# ===================== MATH ENGINE =====================

def mean(d): return sum(d)/len(d) if d else 0

def stdev(d):
    if len(d) < 2: return 0
    m = mean(d)
    return math.sqrt(sum((x-m)**2 for x in d)/len(d))

def ema(data, period):
    if len(data) < period: return []
    k = 2/(period+1)
    res = [mean(data[:period])]
    for p in data[period:]:
        res.append(p*k + res[-1]*(1-k))
    return res

def rsi(closes, period=14):
    if len(closes) < period+1: return 50
    gains = [max(closes[i]-closes[i-1], 0) for i in range(1, len(closes))]
    losses = [max(closes[i-1]-closes[i], 0) for i in range(1, len(closes))]
    ag = mean(gains[-period:])
    al = mean(losses[-period:])
    if al == 0: return 100
    return 100 - (100/(1+ag/al))

def macd(closes):
    if len(closes) < 35: return 0, 0, 0
    ef = ema(closes, 12)
    es = ema(closes, 26)
    n = min(len(ef), len(es))
    ml = [ef[-(n-i)] - es[-(n-i)] for i in range(n)]
    if len(ml) < 9: return 0, 0, 0
    sl = ema(ml, 9)
    if not sl: return 0, 0, 0
    return ml[-1], sl[-1], ml[-1]-sl[-1]

def bollinger(closes, period=20):
    if len(closes) < period: return 0, 0, 0
    r = closes[-period:]
    m = mean(r); s = stdev(r)
    return m+2*s, m, m-2*s

def stoch(highs, lows, closes, k=14):
    if len(closes) < k: return 50, 50
    h = max(highs[-k:]); l = min(lows[-k:])
    if h == l: return 50, 50
    sk = 100*(closes[-1]-l)/(h-l)
    vals = []
    for i in range(min(3, len(closes))):
        hi = max(highs[-(k+i):len(highs)-i] or highs)
        li = min(lows[-(k+i):len(lows)-i] or lows)
        if hi != li:
            vals.append(100*(closes[-(i+1)]-li)/(hi-li))
    sd = mean(vals) if vals else sk
    return sk, sd

def williams_r(highs, lows, closes, period=14):
    if len(closes) < period: return -50
    h = max(highs[-period:]); l = min(lows[-period:])
    if h == l: return -50
    return -100*(h-closes[-1])/(h-l)

def cci(highs, lows, closes, period=20):
    if len(closes) < period: return 0
    tp = [(highs[i]+lows[i]+closes[i])/3 for i in range(len(closes))]
    tp_slice = tp[-period:]
    m = mean(tp_slice)
    md = mean([abs(x-m) for x in tp_slice])
    if md == 0: return 0
    return (tp[-1]-m)/(0.015*md)

def atr(highs, lows, closes, period=14):
    if len(closes) < 2: return 0
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i]-lows[i],
                 abs(highs[i]-closes[i-1]),
                 abs(lows[i]-closes[i-1]))
        trs.append(tr)
    return mean(trs[-period:]) if trs else 0

def momentum(closes, period=10):
    if len(closes) <= period: return 0
    return closes[-1] - closes[-(period+1)]

def volume_trend(vols, period=10):
    if not vols or sum(vols) == 0: return "normal", 1.0
    avg = mean(vols[-period:]) if len(vols) >= period else mean(vols)
    last = vols[-1]
    ratio = last/avg if avg > 0 else 1.0
    if ratio > 1.5: return "very_high", ratio
    if ratio > 1.2: return "high", ratio
    if ratio < 0.5: return "very_low", ratio
    if ratio < 0.8: return "low", ratio
    return "normal", ratio

def detect_candle_pattern(opens, highs, lows, closes):
    if len(closes) < 3: return None
    o1,h1,l1,c1 = opens[-3],highs[-3],lows[-3],closes[-3]
    o2,h2,l2,c2 = opens[-2],highs[-2],lows[-2],closes[-2]
    o3,h3,l3,c3 = opens[-1],highs[-1],lows[-1],closes[-1]

    body1 = abs(c1-o1); body2 = abs(c2-o2); body3 = abs(c3-o3)
    range1 = h1-l1 if h1>l1 else 0.0001
    range3 = h3-l3 if h3>l3 else 0.0001

    # Doji
    if body3 < range3*0.1:
        return ("DOJI", "NEUTRAL", 5)

    # Hammer
    lower_shadow = min(o3,c3)-l3
    upper_shadow = h3-max(o3,c3)
    if lower_shadow > body3*2 and upper_shadow < body3*0.5:
        return ("HAMMER 🔨", "UP", 15)

    # Shooting Star
    if upper_shadow > body3*2 and lower_shadow < body3*0.5:
        return ("SHOOTING STAR ⭐", "DOWN", 15)

    # Engulfing UP
    if c1 < o1 and c3 > o3 and c3 > o1 and o3 < c1:
        return ("ENGULFING صاعد 📈", "UP", 20)

    # Engulfing DOWN
    if c1 > o1 and c3 < o3 and c3 < o1 and o3 > c1:
        return ("ENGULFING هابط 📉", "DOWN", 20)

    # Three White Soldiers
    if c1>o1 and c2>o2 and c3>o3 and c2>c1 and c3>c2:
        return ("THREE SOLDIERS 🪖", "UP", 25)

    # Three Black Crows
    if c1<o1 and c2<o2 and c3<o3 and c2<c1 and c3<c2:
        return ("THREE CROWS 🐦‍⬛", "DOWN", 25)

    return None

def support_resistance(closes, highs, lows, lookback=30):
    if len(closes) < lookback: return None, None
    price = closes[-1]
    recent_h = highs[-lookback:]
    recent_l = lows[-lookback:]
    resistance = max(recent_h)
    support = min(recent_l)
    return support, resistance

def zigzag(closes, threshold=0.003):
    if len(closes) < 10: return None
    direction = None; last = closes[0]; last_type = None; last_idx = 0
    for i in range(1, len(closes)):
        ch = (closes[i]-last)/last if last != 0 else 0
        if direction is None:
            if abs(ch) >= threshold:
                direction = "up" if ch > 0 else "down"
                last = closes[i]; last_idx = i
        elif direction == "up":
            if ch <= -threshold:
                last_type = "high"; direction = "down"
                last = closes[i]; last_idx = i
            elif closes[i] > last:
                last = closes[i]; last_idx = i
        elif direction == "down":
            if ch >= threshold:
                last_type = "low"; direction = "up"
                last = closes[i]; last_idx = i
            elif closes[i] < last:
                last = closes[i]; last_idx = i
    return (last_type, last_idx) if last_type else None

def trend_strength(closes, period=20):
    if len(closes) < period: return "SIDEWAYS"
    e_fast = ema(closes, 5)
    e_slow = ema(closes, period)
    if not e_fast or not e_slow: return "SIDEWAYS"
    diff_pct = (e_fast[-1] - e_slow[-1]) / e_slow[-1] * 100
    if diff_pct > 0.1: return "UPTREND"
    if diff_pct < -0.1: return "DOWNTREND"
    return "SIDEWAYS"

# ===================== DATA FETCH =====================

def fetch(symbol, interval, size=150):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol":symbol,"interval":interval,
              "outputsize":size,"apikey":TWELVE_DATA_KEY,"format":"JSON"}
    try:
        r = requests.get(url, params=params, timeout=15)
        d = r.json()
        if "values" not in d: return None
        v = list(reversed(d["values"]))
        return {
            "closes":  [float(x["close"])  for x in v],
            "opens":   [float(x["open"])   for x in v],
            "highs":   [float(x["high"])   for x in v],
            "lows":    [float(x["low"])    for x in v],
            "volumes": [float(x.get("volume",0)) for x in v],
        }
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return None

# ===================== ANALYSIS ENGINE =====================

def analyze(symbol, interval):
    d = fetch(symbol, interval)
    if not d or len(d["closes"]) < 50:
        return {"error": "لا توجد بيانات كافية"}

    C = d["closes"]; H = d["highs"]; L = d["lows"]
    O = d["opens"]; V = d["volumes"]
    price = C[-1]

    # --- Calculate all indicators ---
    rsi_v    = rsi(C)
    ml,sl,mh = macd(C)
    bu,bm,bl = bollinger(C)
    e5       = ema(C,5)[-1]  if ema(C,5)  else price
    e9       = ema(C,9)[-1]  if ema(C,9)  else price
    e21      = ema(C,21)[-1] if ema(C,21) else price
    e50      = ema(C,50)[-1] if ema(C,50) else price
    stk,std_v= stoch(H,L,C)
    wr       = williams_r(H,L,C)
    cci_v    = cci(H,L,C)
    atr_v    = atr(H,L,C)
    mom      = momentum(C)
    vol_s, vol_r = volume_trend(V)
    candle   = detect_candle_pattern(O,H,L,C)
    sup,res  = support_resistance(C,H,L)
    zz       = zigzag(C)
    trend    = trend_strength(C)

    score = 0; max_score = 0; signals = []

    # 1. RSI (15pts)
    max_score += 15
    if rsi_v < 25:
        score += 15; signals.append(("✅","RSI ذروة بيع قوية جداً","UP"))
    elif rsi_v < 35:
        score += 10; signals.append(("🟢","RSI ذروة بيع","UP"))
    elif rsi_v < 45:
        score += 5;  signals.append(("🟡","RSI منطقة شراء","UP"))
    elif rsi_v > 75:
        score -= 15; signals.append(("✅","RSI ذروة شراء قوية جداً","DOWN"))
    elif rsi_v > 65:
        score -= 10; signals.append(("🔴","RSI ذروة شراء","DOWN"))
    elif rsi_v > 55:
        score -= 5;  signals.append(("🟡","RSI منطقة بيع","DOWN"))
    else:
        signals.append(("⚪","RSI محايد","NEUTRAL"))

    # 2. MACD (15pts)
    max_score += 15
    if ml > sl and mh > 0:
        score += 15; signals.append(("✅","MACD تقاطع صاعد قوي","UP"))
    elif ml > sl:
        score += 8;  signals.append(("🟢","MACD فوق الإشارة","UP"))
    elif ml < sl and mh < 0:
        score -= 15; signals.append(("✅","MACD تقاطع هابط قوي","DOWN"))
    elif ml < sl:
        score -= 8;  signals.append(("🔴","MACD تحت الإشارة","DOWN"))

    # 3. Bollinger (10pts)
    max_score += 10
    bb_range = bu-bl
    bb_pos = (price-bl)/bb_range if bb_range > 0 else 0.5
    if bb_pos < 0.1:
        score += 10; signals.append(("✅","البولينجر: ضغط شراء قوي","UP"))
    elif bb_pos < 0.25:
        score += 6;  signals.append(("🟢","البولينجر: قريب من الدعم","UP"))
    elif bb_pos > 0.9:
        score -= 10; signals.append(("✅","البولينجر: ضغط بيع قوي","DOWN"))
    elif bb_pos > 0.75:
        score -= 6;  signals.append(("🔴","البولينجر: قريب من المقاومة","DOWN"))
    else:
        signals.append(("⚪","البولينجر: منطقة محايدة","NEUTRAL"))

    # 4. EMA Alignment (10pts)
    max_score += 10
    if price > e5 > e9 > e21 > e50:
        score += 10; signals.append(("✅","EMA: تراص صاعد كامل 🚀","UP"))
    elif price > e9 > e21:
        score += 6;  signals.append(("🟢","EMA: اتجاه صاعد","UP"))
    elif price < e5 < e9 < e21 < e50:
        score -= 10; signals.append(("✅","EMA: تراص هابط كامل 📉","DOWN"))
    elif price < e9 < e21:
        score -= 6;  signals.append(("🔴","EMA: اتجاه هابط","DOWN"))
    else:
        signals.append(("⚪","EMA: محايد","NEUTRAL"))

    # 5. Stochastic (8pts)
    max_score += 8
    if stk < 15 and std_v < 20:
        score += 8; signals.append(("✅","Stochastic: ذروة بيع قوية","UP"))
    elif stk < 25:
        score += 5; signals.append(("🟢","Stochastic: منطقة شراء","UP"))
    elif stk > 85 and std_v > 80:
        score -= 8; signals.append(("✅","Stochastic: ذروة شراء قوية","DOWN"))
    elif stk > 75:
        score -= 5; signals.append(("🔴","Stochastic: منطقة بيع","DOWN"))
    else:
        signals.append(("⚪","Stochastic: محايد","NEUTRAL"))

    # 6. Williams %R (8pts)
    max_score += 8
    if wr < -85:
        score += 8; signals.append(("✅","Williams %R: ذروة بيع","UP"))
    elif wr < -70:
        score += 4; signals.append(("🟢","Williams %R: منطقة شراء","UP"))
    elif wr > -15:
        score -= 8; signals.append(("✅","Williams %R: ذروة شراء","DOWN"))
    elif wr > -30:
        score -= 4; signals.append(("🔴","Williams %R: منطقة بيع","DOWN"))
    else:
        signals.append(("⚪","Williams %R: محايد","NEUTRAL"))

    # 7. CCI (8pts)
    max_score += 8
    if cci_v < -150:
        score += 8; signals.append(("✅","CCI: ذروة بيع قوية","UP"))
    elif cci_v < -100:
        score += 4; signals.append(("🟢","CCI: منطقة شراء","UP"))
    elif cci_v > 150:
        score -= 8; signals.append(("✅","CCI: ذروة شراء قوية","DOWN"))
    elif cci_v > 100:
        score -= 4; signals.append(("🔴","CCI: منطقة بيع","DOWN"))
    else:
        signals.append(("⚪","CCI: محايد","NEUTRAL"))

    # 8. Momentum (6pts)
    max_score += 6
    if mom > 0:
        score += 6; signals.append(("🟢",f"Momentum: صاعد +{round(mom,5)}","UP"))
    elif mom < 0:
        score -= 6; signals.append(("🔴",f"Momentum: هابط {round(mom,5)}","DOWN"))
    else:
        signals.append(("⚪","Momentum: محايد","NEUTRAL"))

    # 9. Volume (8pts)
    max_score += 8
    if vol_s in ("high","very_high"):
        if score > 0:
            score += 8; signals.append(("✅",f"Volume: مرتفع {round(vol_r,1)}x يدعم الصعود","UP"))
        else:
            score -= 8; signals.append(("✅",f"Volume: مرتفع {round(vol_r,1)}x يدعم الهبوط","DOWN"))
    elif vol_s in ("low","very_low"):
        signals.append(("⚠️","Volume: منخفض — إشارة ضعيفة","NEUTRAL"))
    else:
        signals.append(("⚪","Volume: طبيعي","NEUTRAL"))

    # 10. Candle Pattern (bonus up to 25pts)
    max_score += 25
    if candle:
        name, direction_c, pts = candle
        if direction_c == "UP":
            score += pts; signals.append(("🕯️",f"كاندل: {name}","UP"))
        elif direction_c == "DOWN":
            score -= pts; signals.append(("🕯️",f"كاندل: {name}","DOWN"))
        else:
            signals.append(("🕯️",f"كاندل: {name}","NEUTRAL"))
    else:
        signals.append(("⚪","كاندل: لا نمط واضح","NEUTRAL"))

    # 11. Support/Resistance (5pts)
    max_score += 5
    if sup and res:
        sr_range = res - sup
        if sr_range > 0:
            pos = (price-sup)/sr_range
            if pos < 0.1:
                score += 5; signals.append(("✅","السعر عند الدعم القوي","UP"))
            elif pos > 0.9:
                score -= 5; signals.append(("✅","السعر عند المقاومة القوية","DOWN"))
            else:
                signals.append(("⚪",f"دعم: {round(sup,5)} | مقاومة: {round(res,5)}","NEUTRAL"))

    # 12. ZigZag (6pts)
    max_score += 6
    if zz:
        zt, zi = zz
        recent = len(C) - zi
        if zt == "low" and recent <= 4:
            score += 6; signals.append(("✅","ZigZag: قاع جديد → ارتداد صاعد","UP"))
        elif zt == "high" and recent <= 4:
            score -= 6; signals.append(("✅","ZigZag: قمة جديدة → تراجع هابط","DOWN"))
        else:
            signals.append(("⚪","ZigZag: لا نقطة محورية حديثة","NEUTRAL"))

    # 13. Trend Filter (bonus 5pts)
    max_score += 5
    if trend == "UPTREND" and score > 0:
        score += 5; signals.append(("✅","الاتجاه العام: صاعد 📈","UP"))
    elif trend == "DOWNTREND" and score < 0:
        score -= 5; signals.append(("✅","الاتجاه العام: هابط 📉","DOWN"))
    elif trend == "SIDEWAYS":
        signals.append(("⚠️","الاتجاه العام: عرضي — توخ الحذر","NEUTRAL"))

    # ===================== FINAL =====================
    confidence = abs(score)/max_score*100
    direction = "UP" if score > 0 else "DOWN" if score < 0 else "NEUTRAL"

    if confidence >= 80:
        strength = "قوية جداً ⚡⚡⚡"
        stars = "★★★★★"
    elif confidence >= 65:
        strength = "قوية 🟢"
        stars = "★★★★☆"
    elif confidence >= 50:
        strength = "متوسطة 🟡"
        stars = "★★★☆☆"
    elif confidence >= 35:
        strength = "ضعيفة 🟠"
        stars = "★★☆☆☆"
    else:
        strength = "ضعيفة جداً 🔴"
        stars = "★☆☆☆☆"

    # Volatility check
    atr_pct = (atr_v/price*100) if price > 0 else 0
    volatility = "مرتفع ⚠️" if atr_pct > 0.3 else "طبيعي ✅" if atr_pct > 0.1 else "منخفض 💤"

    return {
        "direction": direction, "confidence": round(confidence,1),
        "strength": strength, "stars": stars, "signals": signals,
        "rsi": round(rsi_v,1), "price": round(price,5),
        "stoch_k": round(stk,1), "wr": round(wr,1),
        "cci": round(cci_v,1), "vol_trend": vol_s,
        "e9": round(e9,5), "e21": round(e21,5),
        "trend": trend, "volatility": volatility,
        "atr": round(atr_v,5), "support": round(sup,5) if sup else 0,
        "resistance": round(res,5) if res else 0,
        "score": score, "max_score": max_score,
    }

# ===================== MESSAGE =====================

def build_msg(symbol, interval, r):
    d = r["direction"]
    if d == "UP":
        arrow = "⬆️"; color = "🟢"; bg = "📗"
    elif d == "DOWN":
        arrow = "⬇️"; color = "🔴"; bg = "📕"
    else:
        arrow = "↔️"; color = "⚪"; bg = "📒"

    conf = r["confidence"]
    filled = int(conf/10)
    bar = "█"*filled + "░"*(10-filled)

    # Filter signals
    up_sigs   = [f"  {e} {desc}" for e,desc,s in r["signals"] if s=="UP"]
    down_sigs = [f"  {e} {desc}" for e,desc,s in r["signals"] if s=="DOWN"]
    warn_sigs = [f"  {e} {desc}" for e,desc,s in r["signals"] if s=="NEUTRAL" and e=="⚠️"]

    up_text   = "\n".join(up_sigs)   if up_sigs   else "  — لا إشارات صاعدة"
    down_text = "\n".join(down_sigs) if down_sigs else "  — لا إشارات هابطة"
    warn_text = ("\n⚠️ *تحذيرات:*\n" + "\n".join(warn_sigs)) if warn_sigs else ""

    # Recommendation
    if conf >= 80 and d != "NEUTRAL":
        rec = f"✅ *إشارة قوية جداً — يمكن الدخول بحذر*"
    elif conf >= 65 and d != "NEUTRAL":
        rec = f"🟢 *إشارة جيدة — انتظر تأكيداً إضافياً*"
    elif conf >= 50 and d != "NEUTRAL":
        rec = f"🟡 *إشارة متوسطة — تداول بحجم صغير*"
    else:
        rec = f"🔴 *إشارة ضعيفة — انتظر فرصة أفضل*"

    return f"""
{bg} *TRADE ALGO BOT — PRO* {bg}

🔷 *الزوج:* `{symbol}` | ⏱ `{interval}`
💰 *السعر:* `{r['price']}`
📊 *الاتجاه:* `{r['trend']}`

━━━━━━━━━━━━━━━━━━━━━━━━━━

{color} *الإشارة: {arrow} {d}*
{r['stars']}
🎯 *الثقة:* `{conf}%`
`[{bar}]`
💪 *القوة:* {r['strength']}

━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 *إشارات الصعود:*
{up_text}

📉 *إشارات الهبوط:*
{down_text}
{warn_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━

🔢 *البيانات:*
• RSI: `{r['rsi']}` | Stoch: `{r['stoch_k']}`
• W%R: `{r['wr']}` | CCI: `{r['cci']}`
• دعم: `{r['support']}` | مقاومة: `{r['resistance']}`
• تذبذب: {r['volatility']}

━━━━━━━━━━━━━━━━━━━━━━━━━━

{rec}

⚠️ _للأغراض التعليمية فقط_
🕐 {datetime.now().strftime('%H:%M:%S')}
""".strip()

# ===================== KEYBOARDS =====================

def pairs_kb():
    rows = []
    for i in range(0, len(PAIRS), 2):
        row = []
        for p in PAIRS[i:i+2]:
            tags = ("🔥" if p["hot"] else "") + (" OTC" if p["otc"] else "")
            row.append(InlineKeyboardButton(f"{p['label']}{tags}", callback_data=f"pair:{p['symbol']}"))
        rows.append(row)
    rows.append([InlineKeyboardButton("🏠 رجوع", callback_data="start")])
    return InlineKeyboardMarkup(rows)

def tf_kb(symbol):
    rows = []
    for tf in TIMEFRAMES:
        rec = " ← موصى به" if tf["rec"] else ""
        rows.append([InlineKeyboardButton(
            f"{tf['emoji']} {tf['label']} — {tf['desc']}{rec}",
            callback_data=f"tf:{symbol}:{tf['key']}"
        )])
    rows.append([InlineKeyboardButton("🔙 الأزواج", callback_data="choose_pair")])
    return InlineKeyboardMarkup(rows)

def signal_kb(symbol, interval):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 تحديث الإشارة", callback_data=f"tf:{symbol}:{interval}")],
        [InlineKeyboardButton("🔀 تغيير الزوج", callback_data="choose_pair"),
         InlineKeyboardButton("⏱ تغيير الفريم", callback_data=f"pair:{symbol}")],
        [InlineKeyboardButton("🏠 الرئيسية", callback_data="start")],
    ])

# ===================== HANDLERS =====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("📊 ابدأ التحليل", callback_data="choose_pair")
    ]])
    text = """
🤖 *TRADE ALGO BOT — PRO VERSION*
_Advanced Multi-Indicator Analysis_

━━━━━━━━━━━━━━━━━━━━━━━━━━

*13 مؤشر متداخل:*
✅ RSI + MACD + Bollinger Bands
✅ EMA 5/9/21/50
✅ Stochastic + Williams %R
✅ CCI + Momentum + Volume
✅ Candle Patterns (7 أنماط)
✅ Support/Resistance + ZigZag
✅ Trend Filter

━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ _للأغراض التعليمية فقط_
_لا تتداول بأكثر مما تقدر على خسارته_
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
        await q.edit_message_text(
            "💱 *اختر زوج العملات:*\n🔥 = الأكثر تداولاً | OTC = خارج السوق",
            reply_markup=pairs_kb(), parse_mode=ParseMode.MARKDOWN)
    elif d.startswith("pair:"):
        symbol = d.split(":")[1]
        await q.edit_message_text(
            f"⏱ *اختر الإطار الزمني:*\n📌 الزوج: `{symbol}`",
            reply_markup=tf_kb(symbol), parse_mode=ParseMode.MARKDOWN)
    elif d.startswith("tf:"):
        _, symbol, interval = d.split(":")
        await q.edit_message_text(
            f"⏳ *جاري التحليل العميق...*\n\n🔍 `{symbol}` | `{interval}`\n\n⚙️ تحليل 13 مؤشر...",
            parse_mode=ParseMode.MARKDOWN)
        result = analyze(symbol, interval)
        if "error" in result:
            await q.edit_message_text(f"❌ {result['error']}", reply_markup=signal_kb(symbol, interval))
        else:
            await q.edit_message_text(
                build_msg(symbol, interval, result),
                reply_markup=signal_kb(symbol, interval),
                parse_mode=ParseMode.MARKDOWN)

# ===================== MAIN =====================

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(cb))
    logger.info("🚀 Trade Algo Bot PRO running...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
