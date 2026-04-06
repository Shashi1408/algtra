# Zerodha Intraday Trading System

Fully automated NSE intraday bot using **Zerodha Kite Connect API**, 8 technical
strategies, and a scikit-learn ML ensemble (Random Forest + Gradient Boosting).

---

## Project Structure

```
zerodha_trader/
├── main.py           ← Entry point & main loop
├── broker.py         ← Zerodha Kite Connect wrapper (auth + orders + data)
├── engine.py         ← Screener, SignalEngine (8 strategies), PositionManager
├── indicators.py     ← All technical indicators (RSI, MACD, BB, VWAP, ADX …)
├── ml_model.py       ← ML training, labelling, and inference layer
├── display.py        ← Coloured CLI dashboard
├── config.py         ← ALL parameters (credentials, risk, indicators, ML)
└── requirements.txt  ← pip dependencies
```

---

## Quick Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure credentials
Edit `config.py`:
```python
ZERODHA_API_KEY    = "xxxxxxxxxxxxxxxx"
ZERODHA_API_SECRET = "xxxxxxxxxxxxxxxx"
ZERODHA_USER_ID    = "AB1234"
```

> Get your API key from https://developers.kite.trade/ → Create App → Kite Connect

### 3. Run (paper/dry-run mode — no real orders)
```bash
python main.py --paper
```

### 4. Run (live mode)
```bash
python main.py
```
On first run a browser opens for Zerodha login. Copy the `request_token` from
the redirect URL and paste it when prompted. The token is saved automatically.

---

## How It Works

### Stock Screening  (every 20 min)
1. Pulls **top gainers + high-volume** stocks from NSE via Kite quote API
2. Filters by price range (₹50–₹10,000), volume spike (≥1.8× avg), ATR%

### Signal Engine  (every 60 sec)
Scores each candidate 0–10 across 8 rule-based strategies + ML:

| # | Strategy | Condition |
|---|----------|-----------|
| 1 | RSI Reversal | RSI < 32 (buy) / > 68 (sell) |
| 2 | EMA Crossover | EMA9/21 cross + EMA50 trend filter |
| 3 | MACD Histogram | Expanding in direction |
| 4 | Bollinger Bands | Price at/beyond band |
| 5 | VWAP Position | Price vs intraday VWAP |
| 6 | Stochastic | K-line crossover from OB/OS zone |
| 7 | ADX Strength | ADX ≥ 25 with DI confirmation |
| 8 | OBV Slope | Volume pressure direction |
| ML | Ensemble | RF + GBM + LR soft-vote probability |

Signal fires if **score ≥ 4** (configurable in `config.py`).

### Risk Management
- **Position sizing**: ATR-based (risk = 1.5% of capital per trade)
- **Stop-loss**: 1.5× ATR below/above entry
- **Target**: 2× stop-loss distance (2:1 risk-reward)
- **Trailing stop**: ratchets toward price as trade moves in profit
- **Max positions**: 5 concurrent
- **Daily loss limit**: auto-shutdown at 3% drawdown
- **Square-off**: all positions closed by 15:10 IST

### Order Execution
- Uses Kite `MIS` (intraday) product
- Bracket orders with SL+TP when available
- Fallback to regular MARKET orders

### ML Model
- Labels historical candles automatically (BUY/SELL/HOLD based on forward returns)
- Trains `RandomForest + GradientBoosting + LogisticRegression` ensemble
- Retrains every 7 days on fresh data
- Prediction threshold: 62% confidence required to boost a signal

---

## Configuration Reference (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAPITAL` | 500000 | ₹ trading capital |
| `MAX_OPEN_POSITIONS` | 5 | Max concurrent trades |
| `STOP_LOSS_PCT` | 1.2 | SL % from entry |
| `TARGET_PCT` | 2.4 | TP % from entry |
| `TRAILING_SL_PCT` | 0.8 | Trailing SL % |
| `MAX_DAILY_LOSS_PCT` | 3.0 | Auto-shutdown threshold |
| `SIGNAL_THRESHOLD` | 4 | Min score to trade |
| `POLL_INTERVAL_SEC` | 60 | Tick frequency |
| `SQUAREOFF_TIME` | 15:10 | EOD force-close |
| `ML_PROBA_THRESHOLD` | 0.62 | ML confidence gate |

---

## ⚠ Risk Warning

Automated algorithmic trading involves **substantial financial risk**.
This system is provided for **educational and research purposes only**.

- Paper-trade (`--paper`) until you fully understand the system's behaviour
- Always monitor the bot during market hours
- Never trade capital you cannot afford to lose
- Past performance of any algorithm does not guarantee future results
