"""
config.py — Central configuration for the Zerodha Intraday Trading System
Edit ZERODHA_* credentials and tweak all parameters here.
"""

# ══════════════════════════════════════════════
#  ZERODHA CREDENTIALS  (fill these in)
# ══════════════════════════════════════════════
ZERODHA_API_KEY    = "your_api_key_here"
ZERODHA_API_SECRET = "your_api_secret_here"
ZERODHA_USER_ID    = "your_user_id_here"

# After first login Kite Connect gives you an access token.
# The system auto-saves it here; you can also paste a known token.
ACCESS_TOKEN_FILE  = "access_token.txt"

# ══════════════════════════════════════════════
#  CAPITAL & RISK
# ══════════════════════════════════════════════
CAPITAL                = 500_000   # ₹ trading capital
MAX_OPEN_POSITIONS     = 5         # concurrent positions
MAX_RISK_PER_TRADE_PCT = 1.5       # % of capital at risk per trade
STOP_LOSS_PCT          = 1.2       # SL below/above entry (%)
TARGET_PCT             = 2.4       # TP above/below entry (%) → 2:1 RR
TRAILING_SL_PCT        = 0.8       # Trailing stop (% from peak)
MAX_DAILY_LOSS_PCT     = 3.0       # Auto-shutdown if daily loss exceeds this

# ══════════════════════════════════════════════
#  SCREENING PARAMETERS
# ══════════════════════════════════════════════
MIN_PRICE              = 50        # ₹
MAX_PRICE              = 10_000    # ₹
MIN_VOLUME_SPIKE       = 1.8       # Current vol vs 20-period avg
MIN_VOLATILITY_PCT     = 0.5       # Min ATR/price % for intraday opportunity
TOP_GAINERS_N          = 20        # Pull top-N gainers from Kite
SCREEN_INTERVAL_MIN    = 20        # Re-screen every N minutes

# ══════════════════════════════════════════════
#  TECHNICAL INDICATOR PARAMS
# ══════════════════════════════════════════════
RSI_PERIOD   = 14
RSI_OB       = 68    # Overbought
RSI_OS       = 32    # Oversold
EMA_FAST     = 9
EMA_SLOW     = 21
EMA_TREND    = 50
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIG     = 9
BB_PERIOD    = 20
BB_STD       = 2.0
ATR_PERIOD   = 14
STOCH_K      = 14
STOCH_D      = 3
ADX_PERIOD   = 14
ADX_TREND    = 25    # ADX above this = trending market

# ══════════════════════════════════════════════
#  ML MODEL
# ══════════════════════════════════════════════
ML_LOOKBACK        = 30    # candles of feature history
ML_PROBA_THRESHOLD = 0.62  # min confidence to act on ML signal
ML_MODEL_FILE      = "ml_model.pkl"
ML_RETRAIN_DAYS    = 7     # retrain model every N calendar days

# ══════════════════════════════════════════════
#  EXECUTION
# ══════════════════════════════════════════════
POLL_INTERVAL_SEC  = 60         # main loop cadence
ORDER_TYPE         = "MARKET"   # MARKET | LIMIT
EXCHANGE           = "NSE"
PRODUCT            = "MIS"      # MIS = intraday margin product
SIGNAL_THRESHOLD   = 4          # out of 8 composite score
SQUAREOFF_TIME     = "15:10"    # Force close all positions at this time

# ══════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════
LOG_FILE           = "trading_log.txt"
LOG_LEVEL          = "INFO"
