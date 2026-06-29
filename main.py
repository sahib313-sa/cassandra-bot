import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import time
import json
from datetime import datetime, timedelta
import os

# ==================== KONFİQURASİYA ====================

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Zonalar (sənin qiymətlərin)
BUY_ZONE_1 = 4063.33
BUY_ZONE_2 = 4041.75
DECISION_POINT = 4028.07
SELL_ZONE = 4091.66

# Fayl: qiymət tarixçəsi
HISTORY_FILE = "price_history.json"

# ==================== QİYMƏT ÇƏK (Gold-API) ====================
def get_xau_price():
    url = "https://api.gold-api.com/price/XAU"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        price = data.get('price') or data.get('USD', {}).get('price')
        if price is None:
            price = data.get('USD')
        return float(price) if price and float(price) > 1 else None
    except Exception as e:
        print(f"[Xəta] Qiymət: {e}")
        return None

# ==================== TARİXÇƏ ====================
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            data = json.load(f)
            return data.get('prices', []), data.get('timestamps', [])
    return [], []

def save_history(prices, timestamps):
    with open(HISTORY_FILE, 'w') as f:
        json.dump({'prices': prices, 'timestamps': timestamps}, f)

# ==================== İNDİKATORLAR ====================
def calculate_ma(data, period):
    if len(data) < period:
        return None
    return sum(data[-period:]) / period

def calculate_rsi(data, period=14):
    if len(data) < period + 1:
        return 50
    gains, losses = 0, 0
    for i in range(1, period + 1):
        diff = data[-i] - data[-i-1]
        if diff > 0:
            gains += diff
        else:
            losses += abs(diff)
    if losses == 0:
        return 100
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def calculate_atr(data, period=14):
    if len(data) < period + 1:
        return (max(data[-20:]) - min(data[-20:])) / 20
    true_ranges = []
    for i in range(1, len(data)):
        high = max(data[i-1], data[i])
        low = min(data[i-1], data[i])
        true_range = high - low
        true_ranges.append(true_range)
    if len(true_ranges) < period:
        return sum(true_ranges) / len(true_ranges)
    return sum(true_ranges[-period:]) / period

# ==================== SMC ANALİZİ ====================
def detect_order_block(data):
    if len(data) < 4:
        return None, None
    changes = [data[i] - data[i-1] for i in range(1, len(data))]
    if len(changes) < 3:
        return None, None
    if changes[-1] > 0 and changes[-2] > 0 and changes[-3] > 0:
        return "BULLISH_OB", data[-1]
    elif changes[-1] < 0 and changes[-2] < 0 and changes[-3] < 0:
        return "BEARISH_OB", data[-1]
    return None, None

def detect_fvg(data):
    if len(data) < 4:
        return None, None
    # Sadə FVG: son 3 qiymətdə boşluq varsa
    if data[-1] > data[-2] and data[-2] < data[-3]:
        return "BULLISH_FVG", data[-1]
    elif data[-1] < data[-2] and data[-2] > data[-3]:
        return "BEARISH_FVG", data[-1]
    return None, None

def check_haho_half(data):
    if len(data) < 20:
        return "NEUTRAL"
    ma10 = sum(data[-10:]) / 10
    ma20 = sum(data[-20:]) / 20
    haho = ma10 - ma20
    last5 = sum(data[-5:]) / 5
    prev5 = sum(data[-10:-5]) / 5
    half = last5 - prev5
    if haho > 0 and half > 0:
        return "BULLISH"
    elif haho < 0 and half < 0:
        return "BEARISH"
    return "NEUTRAL"

# ==================== MARKET BİAS (Həftəlik/Günlük/H4/H1) ====================
def get_market_bias(data):
    if len(data) < 100:
        return "NEUTRAL"
    
    # Həftəlik (son 7 gün)
    weekly = data[-7:]
    weekly_ma = sum(weekly) / len(weekly)
    
    # Günlük (son 24 saat)
    daily = data[-24:]
    daily_ma = sum(daily) / len(daily)
    
    # H4 (son 6 qiymət)
    h4 = data[-6:]
    h4_ma = sum(h4) / len(h4)
    
    # H1 (son 24 qiymət)
    h1 = data[-24:]
    h1_ma = sum(h1) / len(h1)
    
    bullish_votes = 0
    bearish_votes = 0
    
    if data[-1] > weekly_ma:
        bullish_votes += 1
    else:
        bearish_votes += 1
    
    if data[-1] > daily_ma:
        bullish_votes += 1
    else:
        bearish_votes += 1
    
    if data[-1] > h4_ma:
        bullish_votes += 1
    else:
        bearish_votes += 1
    
    if data[-1] > h1_ma:
        bullish_votes += 1
    else:
        bearish_votes += 1
    
    return "BUY" if bullish_votes > bearish_votes else "SELL" if bearish_votes > bullish_votes else "NEUTRAL"

# ==================== SİQNAL GENERATOR ====================
def generate_signal(price, data):
    if price is None or len(data) < 50:
        return None
    
    # 1. MA50/100 dəstək/dirənc
    ma50 = calculate_ma(data, 50)
    ma100 = calculate_ma(data, 100)
    if ma50 is None or ma100 is None:
        return None
    
    # 2. RSI filtr
    rsi = calculate_rsi(data, 14)
    if rsi > 70 or rsi < 30:
        return None
    
    # 3. Haho/Half
    haho = check_haho_half(data)
    if haho == "NEUTRAL":
        return None
    
    # 4. Order Block / FVG
    ob_type, _ = detect_order_block(data)
    fvg_type, _ = detect_fvg(data)
    if ob_type is None and fvg_type is None:
        return None
    
    # 5. Market Bias
    bias = get_market_bias(data)
    if bias == "NEUTRAL":
        return None
    
    # 6. Bias ilə OB uyğunluğu
    if bias == "BUY" and ob_type != "BULLISH_OB":
        return None
    if bias == "SELL" and ob_type != "BEARISH_OB":
        return None
    
    # 7. Gövdə bağlanışı (H2, H1, M30, M15)
    now = datetime.now()
    if not (now.minute % 15 == 0 or now.minute % 30 == 0 or now.minute % 60 == 0 or now.minute % 120 == 0):
        return None
    
    # 8. ATR
    atr = calculate_atr(data, 14)
    
    # 9. Entry, SL, TP
    if bias == "BUY":
        entry = price
        sl = entry - atr * 1.5
        tp1 = entry + atr * 1.5
        tp2 = entry + atr * 3.0
        tp3 = entry + atr * 4.5
    else:
        entry = price
        sl = entry + atr * 1.5
        tp1 = entry - atr * 1.5
        tp2 = entry - atr * 3.0
        tp3 = entry - atr * 4.5
    
    # 10. RR seçimi (konfluensiya)
    if abs(rsi - 50) < 10:
        rr, tp = 3, tp3
    elif abs(rsi - 50) < 15:
        rr, tp = 2, tp2
    else:
        rr, tp = 1.5, tp1
    
    # 11. Yalnız zonaya iynə atıbsa
    nearest_zone = min([BUY_ZONE_1, BUY_ZONE_2, DECISION_POINT, SELL_ZONE], key=lambda z: abs(z - price))
    if abs(price - nearest_zone) / price > 0.005:
        return None
    
    return {
        'bias': bias,
        'entry': entry,
        'sl': sl,
        'tp': tp,
        'rr': rr,
        'rsi': rsi,
        'ma50': ma50,
        'ma100': ma100,
        'ob_type': ob_type,
        'fvg_type': fvg_type,
        'haho': haho,
        'zone': nearest_zone,
        'confidence': 60 + abs(rsi - 50) / 5
    }

# ==================== QRAFİK ====================
def create_chart(prices, timestamps, signal):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [3, 1]})
    
    if len(prices) > 100:
        prices = prices[-100:]
        timestamps = timestamps[-100:]
    
    ax1.plot(timestamps, prices, color='black', linewidth=1.5, label='XAUUSD')
    
    if len(prices) >= 50:
        ma50 = pd.Series(prices).rolling(50).mean()
        ax1.plot(timestamps, ma50, label='MA50', color='orange', linestyle='--')
    if len(prices) >= 100:
        ma100 = pd.Series(prices).rolling(100).mean()
        ax1.plot(timestamps, ma100, label='MA100', color='purple', linestyle='--')
    
    if signal:
        ax1.axhline(signal['entry'], color='green', linestyle='-', linewidth=2, label=f"Entry {signal['entry']:.2f}")
        ax1.axhline(signal['sl'], color='red', linestyle='--', linewidth=2, label=f"SL {signal['sl']:.2f}")
        ax1.axhline(signal['tp'], color='blue', linestyle='--', linewidth=2, label=f"TP {signal['tp']:.2f}")
        
        info = f"Bias: {signal['bias']} | RR: 1:{signal['rr']} | RSI: {signal['rsi']:.1f}"
        ax1.text(0.02, 0.98, info, transform=ax1.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Zona
        ax1.axhspan(signal['zone'] - 2, signal['zone'] + 2, alpha=0.2, color='green' if signal['bias'] == 'BUY' else 'red')
    
    ax1.set_title(f"XAUUSD - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    if len(prices) >= 14:
        rsi = pd.Series(prices).pct_change().rolling(14).apply(
            lambda x: 100 - (100 / (1 + (x[x>0].mean() / abs(x[x<0].mean()) if x[x<0].mean() != 0 else 1)))
        )
        ax2.plot(timestamps, rsi, color='blue', linewidth=1.5)
        ax2.axhline(70, color='red', linestyle='--')
        ax2.axhline(30, color='green', linestyle='--')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Vaxt')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    return buf

# ==================== TELEGRAM ====================
def send_telegram_photo(photo, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    files = {'photo': ('chart.png', photo, 'image/png')}
    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
    try:
        r = requests.post(url, files=files, data=data, timeout=30)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"[Xəta] Şəkil göndərmə: {e}")
        return False

# ==================== ƏSAS DÖVR ====================
def main():
    print("Bot işə başladı...")
    prices, timestamps = load_history()
    daily_signals = 0
    last_signal_date = None
    
    while True:
        try:
            now = datetime.now()
            
            # Gündəlik limit
            today = now.date()
            if last_signal_date != today:
                daily_signals = 0
                last_signal_date = today
            
            print(f"[{now}] Qiymət çəkilir...")
            price = get_xau_price()
            if price is None:
                print("Qiymət alınmadı, 60s gözləyirik...")
                time.sleep(60)
                continue
            
            prices.append(price)
            timestamps.append(now.isoformat())
            if len(prices) > 200:
                prices = prices[-200:]
                timestamps = timestamps[-200:]
            save_history(prices, timestamps)
            
            # Siqnal yarat
            signal = generate_signal(price, prices)
            
            # Gündə 2-3 siqnal limiti
            if signal and daily_signals < 3:
                daily_signals += 1
                caption = f"""📊 **XAUUSD SİQNALI**

📈 **İstiqamət:** {signal['bias']}
📍 **Entry (Limit):** {signal['entry']:.2f}
🔴 **Stop Loss:** {signal['sl']:.2f}
🔵 **Take Profit:** {signal['tp']:.2f} (1:{signal['rr']})
📊 **RSI:** {signal['rsi']:.1f}
📈 **MA50:** {signal['ma50']:.2f}
📈 **MA100:** {signal['ma100']:.2f}
🔹 **Order Block:** {signal['ob_type']}
🔹 **FVG:** {signal['fvg_type']}
🔹 **Haho/Half:** {signal['haho']}
📍 **Zona:** {signal['zone']:.2f}
🎯 **Etibarlılıq:** {signal['confidence']:.1f}%

⚠️ Yatırım tövsiyəsi deyil. Riskinizi özünüz idarə edin."""
                
                chart = create_chart(prices, timestamps, signal)
                send_telegram_photo(chart, caption)
                print(f"[{now}] ✅ SİQNAL GÖNDƏRİLDİ! Entry: {signal['entry']:.2f}")
            else:
                print(f"[{now}] ❌ Siqnal yoxdur. Qiymət: {price:.2f}")
            
            # Həftəsonu analiz
            if now.weekday() in [5, 6]:  # Şənbə və ya Bazar
                msg = f"📊 **Həftəsonu Analiz**\n\nQiymət: {price:.2f}\nBias: {get_market_bias(prices)}\nYaxın zona: {min([BUY_ZONE_1, BUY_ZONE_2, DECISION_POINT, SELL_ZONE], key=lambda z: abs(z - price)):.2f}"
                send_telegram_photo(None, msg)  # Sadəcə mətn göndər
            
            time.sleep(3600)
            
        except Exception as e:
            print(f"[Dövr xətası] {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
from flask import Flask
from threading import Thread

app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is running!"

def run():
    app.run(host='0.0.0.0', port=10000)

# Botu işə salmazdan əvvəl bu thread-i başladaq
t = Thread(target=run)
t.start()
