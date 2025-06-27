import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as animation
from sklearn.linear_model import LinearRegression
import asyncio
import json
import websockets
import threading
import queue
import requests

USD_TO_INR = 86.83  # default fallback rate

def fetch_usd_to_inr():
    url = "https://open.er-api.com/v6/latest/USD"  # Free API, no key required
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if 'rates' in data and 'INR' in data['rates']:
            return data['rates']['INR']
        else:
            print("API response format unexpected:", data)
            return None
    except Exception as e:
        print("Failed to fetch USD to INR rate:", e)
        return None

def update_usd_to_inr_rate():
    global USD_TO_INR
    rate = fetch_usd_to_inr()
    if rate is not None:
        print(f"USD_TO_INR rate: {rate}")
        USD_TO_INR = rate
    else:
        print(f"Using fallback USD_TO_INR rate: {USD_TO_INR}")

update_usd_to_inr_rate()


is_paused=False
tickers = ['btcusdt', 'ethusdt', 'bnbusdt']
predict_seconds = 10
history_length = 60
colors = ['b', 'g', 'r']
pred_colors = ['c', 'lime', 'orange']

binance_url = "wss://stream.binance.com:9443/stream?streams=" + '/'.join(f"{ticker}@ticker" for ticker in tickers)

price_queue = {ticker: queue.Queue() for ticker in tickers}
x_data = []
y_data = {ticker: [] for ticker in tickers}
pred_points = {ticker: [] for ticker in tickers}
initial_price = {}

fig, ax = plt.subplots()
lines = {}
pred_scatters = {}

for i, ticker in enumerate(tickers):
    line, = ax.plot([], [], colors[i], label=ticker.upper())
    lines[ticker] = line
    scat = ax.scatter([], [], color=pred_colors[i], label=f'{ticker.upper()} (Pred)')
    pred_scatters[ticker] = scat

ax.set_title("Live Crypto Prices (INR) + 10s Prediction")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Price (INR)")
ax.set_xlim(0, history_length + predict_seconds)
ax.set_ylim(0, 1)
ax.legend(loc="upper left")
plt.tight_layout()

# Top-right info text per ticker
info_texts = {
    ticker: ax.text(
        0.98, 0.95 - i * 0.1, '',
        transform=ax.transAxes,
        ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    for i, ticker in enumerate(tickers)
}

# Hover annotation setup
annot = ax.annotate(
    "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
    bbox=dict(boxstyle="round", fc="w"),
    arrowprops=dict(arrowstyle="->")
)
annot.set_visible(False)

def update_annot(ticker, artist, ind, is_pred=False):
    x, y = artist.get_offsets().T if is_pred else artist.get_data()
    idx = ind["ind"][0]
    pos = (x[idx], y[idx])
    annot.xy = pos

    if ticker in initial_price:
        actual_inr = (pos[1] - offsets[ticker]) * initial_price[ticker] * USD_TO_INR
    else:
        actual_inr = pos[1]

    label_type = "Predicted" if is_pred else "Actual"
    text = (
        f"{ticker.upper()} ({label_type})\n"
        f"Time: {pos[0]:.0f}s\n"
        f"Price: ₹{actual_inr:,.2f}"
    )
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.9)

def hover(event):
    if event.inaxes == ax:
        for ticker, line in lines.items():
            cont, ind = line.contains(event)
            if cont:
                update_annot(ticker, line, ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return
            
        for ticker, scat in pred_scatters.items():
            cont, ind=scat.contains(event)
            if cont:
                update_annot(ticker, scat, ind, is_pred=True)
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return
    annot.set_visible(False)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

# Binance WebSocket listener
def start_websocket():
    async def listen():
        async with websockets.connect(binance_url) as ws:
            print("✅ Connected to Binance WebSocket")
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                ticker = data['stream'].split('@')[0]
                price_usdt = float(data['data']['c'])
                price_queue[ticker].put(price_usdt)
    asyncio.run(listen())

# Linear prediction
def predict_next(y_vals):
    if len(y_vals) < 10:
        return []
    X = np.array(range(len(y_vals))).reshape(-1, 1)
    y = np.array(y_vals)
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.array(range(len(y_vals), len(y_vals) + predict_seconds)).reshape(-1, 1)
    future_preds = model.predict(future_X)
    future_t = list(range(len(x_data), len(x_data) + predict_seconds))
    return list(zip(future_t, future_preds))

# Offset each line for clarity
offsets = {'btcusdt': 0.0, 'ethusdt': 0.2, 'bnbusdt': 0.4}
frame_count = 0

def animate(frame):
    global frame_count
    if is_paused:
        return list(lines.values())+list(pred_scatters.values())+list(info_texts.values())+[annot]
    frame_count += 1
    x_data.append(frame_count)

    all_y = []

    for ticker in tickers:
        if not price_queue[ticker].empty():
            price = price_queue[ticker].get()
            if ticker not in initial_price:
                initial_price[ticker] = price

            norm_price = price / initial_price[ticker]
            shifted_price = norm_price + offsets[ticker]
            y_data[ticker].append(shifted_price)

        y_vals = y_data[ticker][-history_length:]
        if not y_vals:
            continue

        x_vals = x_data[-len(y_vals):]
        if len(x_vals) != len(y_vals):
            min_len = min(len(x_vals), len(y_vals))
            x_vals = x_vals[-min_len:]
            y_vals = y_vals[-min_len:]

        lines[ticker].set_data(x_vals, y_vals)

        # Predict future
        preds = predict_next([p - offsets[ticker] for p in y_vals])
        pred_points[ticker] = [(t, p + offsets[ticker]) for t, p in preds]


        if pred_points[ticker]:
            px, py = zip(*pred_points[ticker])
            pred_scatters[ticker].set_offsets(np.column_stack((px, py)))
            all_y += list(py)

        # Info box
        if y_vals:
            real_price_inr = (y_vals[-1] - offsets[ticker]) * initial_price[ticker] * USD_TO_INR
            info_texts[ticker].set_text(f"{ticker.upper()}: ₹{real_price_inr:,.2f}")
            all_y += y_vals

    if all_y:
        ax.set_ylim(min(all_y) * 0.995, max(all_y) * 1.005)
        buffer=int(predict_seconds*0.3)
        ax.set_xlim(max(0, frame_count - history_length), frame_count + predict_seconds+buffer)

    return list(lines.values()) + list(pred_scatters.values()) + list(info_texts.values()) + [annot]

def on_key(event):
    global is_paused
    if event.key==' ':
        is_paused=not is_paused
        print("Paused" if is_paused else "Resumed")
fig.canvas.mpl_connect('key_press_event',on_key)

# Start WebSocket in background
ws_thread = threading.Thread(target=start_websocket, daemon=True)
ws_thread.start()

# Start animation
ani = animation(fig, animate, interval=1000)
plt.show()
