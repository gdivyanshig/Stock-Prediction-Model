# 📈 Live Crypto Price & Prediction Dashboard

This Python project provides a real-time animated dashboard of **cryptocurrency prices (converted to INR)** from the Binance WebSocket feed. It features **10-second linear price prediction**, dynamic tooltips, annotations, and real-time exchange rate conversion (USD to INR). It supports live updates for **BTC, ETH, and BNB**.

---

## 💡 Features

- 🔄 Live crypto prices from Binance WebSocket API  
-  Real-time conversion from USD to INR using open exchange rate API  
- 🔮 Predicts prices 10 seconds into the future using linear regression  
- 🎨 Matplotlib animation with smooth updates  
- 🖱️ Hover tooltips showing predicted and actual prices in INR  
- ⏸️ Pause/resume chart updates with the `Spacebar`  
- 📦 Multi-threaded WebSocket + matplotlib animation  

---

## 🧰 Technologies Used

- Python 3.7+  
- `matplotlib`  
- `numpy`  
- `scikit-learn`  
- `requests`, `websockets`, `asyncio`, `threading`  
- Binance WebSocket API  
- Open Exchange Rate API (`https://open.er-api.com`)  

---

## 🚀 Getting Started

### ✅ Prerequisites

Install the required packages using pip:

```bash
pip install numpy matplotlib scikit-learn requests websockets
```

---

## 🔁 USD to INR Conversion

- The app fetches the current USD to INR rate via a free exchange API.
- If the request fails, it falls back to a static rate of **₹86.83**.
- The conversion is applied on top of live Binance price data.

---

## 📉 Cryptocurrencies Tracked

The following crypto pairs are tracked by default:

- **BTC/USDT** → `btcusdt`  
- **ETH/USDT** → `ethusdt`  
- **BNB/USDT** → `bnbusdt`

You can add or remove coins by editing the `tickers` list in the script.

---

## 🖱️ Controls

| Action         | Function                        |
|----------------|---------------------------------|
| Hover cursor   | Show INR price tooltip          |
| `Spacebar`     | Pause/resume chart animation    |

---

> Feel free to ⭐ this project, fork it, or contribute to it. Pull requests are welcome!
