"""
engine.py — Screener, multi-strategy SignalEngine, and PositionManager
"""

import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import config as cfg
from indicators import Indicators
from ml_model import MLSignalModel

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Signal:
    symbol:     str
    action:     str            # BUY | SELL | HOLD
    price:      float
    score:      int            # 0 – MAX_SCORE
    confidence: float = 0.0   # ML probability
    reasons:    list  = field(default_factory=list)
    stop_loss:  float = 0.0
    target:     float = 0.0
    qty:        int   = 0
    strategy:   str   = ""
    timestamp:  str   = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


@dataclass
class Position:
    symbol:     str
    side:       str      # LONG | SHORT
    entry:      float
    qty:        int
    stop_loss:  float
    target:     float
    peak:       float    # for trailing stop
    order_id:   str  = ""
    entry_time: str  = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


# ─────────────────────────────────────────────────────────────────────────────
# SCREENER
# ─────────────────────────────────────────────────────────────────────────────
class Screener:
    """
    Pulls the broad universe from the broker, fetches OHLCV, and filters
    down to high-momentum intraday candidates using:
      • Volume spike  (≥ MIN_VOLUME_SPIKE × 20-period avg)
      • Volatility    (ATR/price ≥ MIN_VOLATILITY_PCT)
      • Price range   (MIN_PRICE – MAX_PRICE)
    """

    def __init__(self, broker):
        self.broker = broker

    def run(self, data_cache: dict) -> list[str]:
        symbols  = self.broker.get_top_movers()
        passed   = []
        log.info(f"[SCREEN] Evaluating {len(symbols)} symbols …")

        for sym in symbols:
            df = data_cache.get(sym)
            if df is None or len(df) < 40:
                continue

            close = df["Close"].squeeze()
            price = float(close.iloc[-1])

            if not (cfg.MIN_PRICE <= price <= cfg.MAX_PRICE):
                continue

            vol_ratio = Indicators.volume_ratio(df, 20).iloc[-1]
            if vol_ratio < cfg.MIN_VOLUME_SPIKE:
                continue

            atr_pct = float(Indicators.atr(df, cfg.ATR_PERIOD).iloc[-1]) / price * 100
            if atr_pct < cfg.MIN_VOLATILITY_PCT:
                continue

            passed.append(sym)
            log.info(f"[SCREEN]  ✓ {sym:<15} price={price:.2f}  "
                     f"vol_ratio={vol_ratio:.2f}  atr%={atr_pct:.2f}")

        log.info(f"[SCREEN] {len(passed)}/{len(symbols)} passed")
        return passed


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL ENGINE  (8 strategies + ML)
# ─────────────────────────────────────────────────────────────────────────────
class SignalEngine:
    """
    Combines 8 rule-based strategies with an ML ensemble into one composite
    score.  Maximum possible rule score = 8.  ML adds/subtracts up to 2 pts.
    Threshold defined in config.SIGNAL_THRESHOLD.

    Strategies:
      1. RSI reversal
      2. EMA crossover (9/21) + trend filter (50)
      3. MACD histogram momentum
      4. Bollinger Band squeeze / breakout
      5. VWAP relative position
      6. Stochastic crossover
      7. ADX trend confirmation
      8. OBV divergence / confirmation
     ML. Random Forest + Gradient Boosting ensemble
    """

    MAX_RULE_SCORE = 8

    def __init__(self):
        self.ml = MLSignalModel()

    def analyse(self, symbol: str, df: pd.DataFrame) -> Signal:
        close = df["Close"].squeeze()
        price = float(close.iloc[-1])

        # ── Compute all indicators ────────────────────────────────────────────
        rsi      = Indicators.rsi(close, cfg.RSI_PERIOD)
        ema9     = Indicators.ema(close, cfg.EMA_FAST)
        ema21    = Indicators.ema(close, cfg.EMA_SLOW)
        ema50    = Indicators.ema(close, cfg.EMA_TREND)
        macd_l, sig_l, hist = Indicators.macd(close, cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIG)
        bb_up, bb_mid, bb_lo, pct_b = Indicators.bollinger(close, cfg.BB_PERIOD, cfg.BB_STD)
        vwap     = Indicators.vwap(df)
        stk, std = Indicators.stochastic(df, cfg.STOCH_K, cfg.STOCH_D)
        adx, plus_di, minus_di = Indicators.adx(df, cfg.ADX_PERIOD)
        obv      = Indicators.obv(df)
        atr      = Indicators.atr(df, cfg.ATR_PERIOD)

        # Latest values
        r         = float(rsi.iloc[-1])
        e9        = float(ema9.iloc[-1]);  e21 = float(ema21.iloc[-1]); e50 = float(ema50.iloc[-1])
        mh        = float(hist.iloc[-1]); mh_prev = float(hist.iloc[-2])
        bbu       = float(bb_up.iloc[-1]); bbl = float(bb_lo.iloc[-1])
        pb        = float(pct_b.iloc[-1])
        vw        = float(vwap.iloc[-1])
        sk        = float(stk.iloc[-1]);  sd = float(std.iloc[-1]); sk_prev = float(stk.iloc[-2])
        ax        = float(adx.iloc[-1]);  pd_val = float(plus_di.iloc[-1]); md_val = float(minus_di.iloc[-1])
        obv_slope = float(obv.diff(5).iloc[-1])
        cur_atr   = float(atr.iloc[-1])

        buy_score  = 0
        sell_score = 0
        buy_why:  list[str] = []
        sell_why: list[str] = []

        # ── 1. RSI reversal ───────────────────────────────────────────────────
        if r < cfg.RSI_OS:
            buy_score  += 1; buy_why.append(f"RSI={r:.1f} oversold")
        elif r > cfg.RSI_OB:
            sell_score += 1; sell_why.append(f"RSI={r:.1f} overbought")

        # ── 2. EMA crossover with trend filter ───────────────────────────────
        ema_bull = e9 > e21 and price > e50
        ema_bear = e9 < e21 and price < e50
        if ema_bull:
            buy_score  += 1; buy_why.append("EMA9>EMA21 + Price>EMA50")
        elif ema_bear:
            sell_score += 1; sell_why.append("EMA9<EMA21 + Price<EMA50")

        # ── 3. MACD histogram expanding ──────────────────────────────────────
        if mh > 0 and mh > mh_prev:
            buy_score  += 1; buy_why.append("MACD histogram bullish expansion")
        elif mh < 0 and mh < mh_prev:
            sell_score += 1; sell_why.append("MACD histogram bearish expansion")

        # ── 4. Bollinger Band ────────────────────────────────────────────────
        if pb < 0.05:
            buy_score  += 1; buy_why.append(f"Price near lower BB (pctB={pb:.2f})")
        elif pb > 0.95:
            sell_score += 1; sell_why.append(f"Price near upper BB (pctB={pb:.2f})")

        # ── 5. VWAP position ─────────────────────────────────────────────────
        if price > vw * 1.001:
            buy_score  += 1; buy_why.append(f"Price above VWAP ({vw:.2f})")
        elif price < vw * 0.999:
            sell_score += 1; sell_why.append(f"Price below VWAP ({vw:.2f})")

        # ── 6. Stochastic crossover ───────────────────────────────────────────
        if sk < 25 and sk > sd and sk_prev <= sd:
            buy_score  += 1; buy_why.append(f"Stoch K({sk:.1f}) crossing up from OS")
        elif sk > 75 and sk < sd:
            sell_score += 1; sell_why.append(f"Stoch K({sk:.1f}) overbought cross-down")

        # ── 7. ADX trend confirmation ─────────────────────────────────────────
        if ax >= cfg.ADX_TREND:
            if pd_val > md_val:
                buy_score  += 1; buy_why.append(f"ADX={ax:.1f} +DI>{md_val:.1f} trend up")
            else:
                sell_score += 1; sell_why.append(f"ADX={ax:.1f} -DI>{pd_val:.1f} trend down")

        # ── 8. OBV confirmation ───────────────────────────────────────────────
        if obv_slope > 0:
            buy_score  += 1; buy_why.append("OBV rising (buying pressure)")
        elif obv_slope < 0:
            sell_score += 1; sell_why.append("OBV falling (selling pressure)")

        # ── ML layer ─────────────────────────────────────────────────────────
        ml_action, ml_conf = self.ml.predict(df)
        ml_boost = 0
        if ml_action == "BUY"  and ml_conf >= cfg.ML_PROBA_THRESHOLD:
            ml_boost = 2; buy_why.append(f"ML={ml_action} conf={ml_conf:.0%}")
        elif ml_action == "SELL" and ml_conf >= cfg.ML_PROBA_THRESHOLD:
            ml_boost = -2; sell_why.append(f"ML={ml_action} conf={ml_conf:.0%}")

        final_buy  = buy_score  + (ml_boost if ml_boost > 0 else 0)
        final_sell = sell_score + (-ml_boost if ml_boost < 0 else 0)

        # ── Risk sizing (ATR-based) ───────────────────────────────────────────
        sl_pts  = max(cur_atr * 1.5, price * cfg.STOP_LOSS_PCT / 100)
        tp_pts  = sl_pts * 2.0   # minimum 2:1 RR
        sl_long = round(price - sl_pts, 2)
        tp_long = round(price + tp_pts, 2)
        risk_amt = cfg.CAPITAL * cfg.MAX_RISK_PER_TRADE_PCT / 100
        qty     = max(1, int(risk_amt / sl_pts))

        # ── Decision ──────────────────────────────────────────────────────────
        threshold = cfg.SIGNAL_THRESHOLD
        if final_buy >= threshold and final_buy > final_sell:
            return Signal(symbol, "BUY", price, final_buy, ml_conf, buy_why,
                          sl_long, tp_long, qty, "COMPOSITE_BUY")
        elif final_sell >= threshold and final_sell > final_buy:
            sl_short = round(price + sl_pts, 2)
            tp_short = round(price - tp_pts, 2)
            return Signal(symbol, "SELL", price, final_sell, ml_conf, sell_why,
                          sl_short, tp_short, qty, "COMPOSITE_SELL")
        else:
            score = max(final_buy, final_sell)
            reasons = buy_why if final_buy >= final_sell else sell_why
            return Signal(symbol, "HOLD", price, score, ml_conf, reasons)


# ─────────────────────────────────────────────────────────────────────────────
# POSITION MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class PositionManager:
    """
    Tracks open paper/live positions.
    Implements trailing stop-loss by updating peak price on each tick.
    Interfaces with the broker to place real orders when not in dry-run mode.
    """

    def __init__(self, broker):
        self.broker    = broker
        self.positions : dict[str, Position] = {}
        self.realised_pnl : float = 0.0
        self.trade_log : list[dict] = []
        self.daily_loss: float = 0.0

    def max_positions_reached(self) -> bool:
        return len(self.positions) >= cfg.MAX_OPEN_POSITIONS

    def is_daily_loss_limit_hit(self) -> bool:
        limit = cfg.CAPITAL * cfg.MAX_DAILY_LOSS_PCT / 100
        return self.daily_loss >= limit

    def open(self, signal: Signal) -> bool:
        if signal.symbol in self.positions:
            return False
        if self.max_positions_reached():
            log.info(f"[PM] Max positions ({cfg.MAX_OPEN_POSITIONS}) reached. Skipping {signal.symbol}")
            return False
        if self.is_daily_loss_limit_hit():
            log.warning("[PM] Daily loss limit hit. No new trades.")
            return False

        txn = "BUY" if signal.action == "BUY" else "SELL"
        order_id = self.broker.place_order(signal.symbol, txn, signal.qty)
        if order_id is None:
            return False

        side = "LONG" if signal.action == "BUY" else "SHORT"
        self.positions[signal.symbol] = Position(
            symbol=signal.symbol, side=side,
            entry=signal.price, qty=signal.qty,
            stop_loss=signal.stop_loss, target=signal.target,
            peak=signal.price, order_id=order_id,
        )
        log.info(f"[PM] ▶ OPEN {side:<5} {signal.symbol:<15} "
                 f"@ ₹{signal.price:.2f}  SL=₹{signal.stop_loss:.2f}  "
                 f"TP=₹{signal.target:.2f}  Qty={signal.qty}  "
                 f"OID={order_id}")
        return True

    def update_trailing_stop(self, symbol: str, current_price: float):
        """Ratchet stop-loss up (for LONG) or down (for SHORT) as price moves favourably."""
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        trail_dist = pos.entry * cfg.TRAILING_SL_PCT / 100

        if pos.side == "LONG" and current_price > pos.peak:
            pos.peak      = current_price
            new_sl        = round(current_price - trail_dist, 2)
            if new_sl > pos.stop_loss:
                pos.stop_loss = new_sl
        elif pos.side == "SHORT" and current_price < pos.peak:
            pos.peak      = current_price
            new_sl        = round(current_price + trail_dist, 2)
            if new_sl < pos.stop_loss:
                pos.stop_loss = new_sl

    def check_exits(self, symbol: str, current_price: float) -> Optional[str]:
        """Returns exit reason string if position was closed, else None."""
        if symbol not in self.positions:
            return None
        pos = self.positions[symbol]
        self.update_trailing_stop(symbol, current_price)
        exit_reason = None

        if pos.side == "LONG":
            if current_price <= pos.stop_loss:
                exit_reason = "STOP-LOSS"
            elif current_price >= pos.target:
                exit_reason = "TARGET"
        else:
            if current_price >= pos.stop_loss:
                exit_reason = "STOP-LOSS"
            elif current_price <= pos.target:
                exit_reason = "TARGET"

        if exit_reason:
            self._close(symbol, current_price, exit_reason)
        return exit_reason

    def _close(self, symbol: str, price: float, reason: str):
        pos = self.positions[symbol]
        pnl = ((price - pos.entry) * pos.qty
               if pos.side == "LONG"
               else (pos.entry - price) * pos.qty)

        # Place closing order on broker
        close_txn = "SELL" if pos.side == "LONG" else "BUY"
        self.broker.place_order(symbol, close_txn, pos.qty)

        self.realised_pnl += pnl
        if pnl < 0:
            self.daily_loss += abs(pnl)

        log.info(f"[PM] ◼ CLOSE {pos.side:<5} {symbol:<15} "
                 f"@ ₹{price:.2f}  P&L={pnl:+.2f}  [{reason}]")
        self.trade_log.append({
            "symbol": symbol, "side": pos.side,
            "entry": pos.entry, "exit": price,
            "qty": pos.qty, "pnl": round(pnl, 2),
            "reason": reason,
            "time": datetime.now().strftime("%H:%M:%S"),
        })
        del self.positions[symbol]

    def squareoff_all(self, current_prices: dict):
        """Force-close all open positions (called at EOD)."""
        log.info("[PM] ⚡ Square-off all positions")
        for sym in list(self.positions.keys()):
            price = current_prices.get(sym, self.positions[sym].entry)
            self._close(sym, price, "SQUAREOFF")

    def unrealised_pnl(self, current_prices: dict) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            cp = current_prices.get(sym, pos.entry)
            pnl = ((cp - pos.entry) * pos.qty
                   if pos.side == "LONG"
                   else (pos.entry - cp) * pos.qty)
            total += pnl
        return total
