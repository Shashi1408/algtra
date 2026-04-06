"""
indicators.py — Pure-numpy/pandas technical indicator library
No external TA library required; all computed from scratch.
"""

import numpy as np
import pandas as pd


class Indicators:

    # ── Trend ─────────────────────────────────────────────────────────────────
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(period).mean()

    @staticmethod
    def macd(close: pd.Series, fast=12, slow=26, signal=9):
        fast_ema    = Indicators.ema(close, fast)
        slow_ema    = Indicators.ema(close, slow)
        macd_line   = fast_ema - slow_ema
        signal_line = Indicators.ema(macd_line, signal)
        histogram   = macd_line - signal_line
        return macd_line, signal_line, histogram

    # ── Momentum ──────────────────────────────────────────────────────────────
    @staticmethod
    def rsi(close: pd.Series, period=14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def stochastic(df: pd.DataFrame, k=14, d=3):
        low_min  = df["Low"].rolling(k).min()
        high_max = df["High"].rolling(k).max()
        k_line   = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
        d_line   = k_line.rolling(d).mean()
        return k_line, d_line

    @staticmethod
    def cci(df: pd.DataFrame, period=20) -> pd.Series:
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        mean    = typical.rolling(period).mean()
        mad     = typical.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical - mean) / (0.015 * mad.replace(0, np.nan))

    # ── Volatility ────────────────────────────────────────────────────────────
    @staticmethod
    def bollinger(close: pd.Series, period=20, std=2.0):
        mid   = close.rolling(period).mean()
        sigma = close.rolling(period).std()
        upper = mid + std * sigma
        lower = mid - std * sigma
        pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
        return upper, mid, lower, pct_b

    @staticmethod
    def atr(df: pd.DataFrame, period=14) -> pd.Series:
        hl  = df["High"] - df["Low"]
        hpc = (df["High"] - df["Close"].shift()).abs()
        lpc = (df["Low"]  - df["Close"].shift()).abs()
        tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def keltner(df: pd.DataFrame, ema_period=20, atr_period=10, mult=2.0):
        mid   = Indicators.ema(df["Close"], ema_period)
        atr   = Indicators.atr(df, atr_period)
        upper = mid + mult * atr
        lower = mid - mult * atr
        return upper, mid, lower

    # ── Volume ────────────────────────────────────────────────────────────────
    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        return (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        direction = np.sign(df["Close"].diff()).fillna(0)
        return (direction * df["Volume"]).cumsum()

    @staticmethod
    def volume_ratio(df: pd.DataFrame, period=20) -> pd.Series:
        """Current volume vs rolling mean."""
        return df["Volume"] / df["Volume"].rolling(period).mean()

    # ── Trend strength ────────────────────────────────────────────────────────
    @staticmethod
    def adx(df: pd.DataFrame, period=14):
        """Returns ADX, +DI, -DI."""
        up   = df["High"].diff()
        down = -df["Low"].diff()
        plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr = Indicators.atr(df, period)
        plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(span=period, adjust=False).mean() / tr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / tr
        dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx, plus_di, minus_di

    # ── Support / Resistance ──────────────────────────────────────────────────
    @staticmethod
    def pivot_points(df: pd.DataFrame):
        """Classic pivot point levels from previous candle."""
        prev = df.iloc[-2]
        P  = (prev["High"] + prev["Low"] + prev["Close"]) / 3
        R1 = 2 * P - prev["Low"]
        S1 = 2 * P - prev["High"]
        R2 = P + (prev["High"] - prev["Low"])
        S2 = P - (prev["High"] - prev["Low"])
        return {"P": P, "R1": R1, "R2": R2, "S1": S1, "S2": S2}

    # ── Composite feature vector for ML ──────────────────────────────────────
    @staticmethod
    def feature_vector(df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a rich feature matrix (one row per candle) for ML training/inference.
        All features are normalised or bounded to reduce scale sensitivity.
        """
        close = df["Close"]
        feat  = pd.DataFrame(index=df.index)

        feat["rsi"]          = Indicators.rsi(close, 14)
        feat["rsi_fast"]     = Indicators.rsi(close, 7)
        feat["stoch_k"], feat["stoch_d"] = Indicators.stochastic(df)
        feat["cci"]          = Indicators.cci(df)
        macd_l, sig_l, hist  = Indicators.macd(close)
        feat["macd_hist"]    = hist
        feat["macd_cross"]   = (macd_l - sig_l).apply(np.sign)

        ema9  = Indicators.ema(close, 9)
        ema21 = Indicators.ema(close, 21)
        ema50 = Indicators.ema(close, 50)
        feat["ema9_21_cross"]  = (ema9 - ema21).apply(np.sign)
        feat["ema21_50_cross"] = (ema21 - ema50).apply(np.sign)
        feat["price_vs_ema9"]  = (close - ema9) / close
        feat["price_vs_ema50"] = (close - ema50) / close

        bb_up, bb_mid, bb_lo, pct_b = Indicators.bollinger(close)
        feat["pct_b"]        = pct_b
        feat["bb_width"]     = (bb_up - bb_lo) / bb_mid

        atr = Indicators.atr(df)
        feat["atr_pct"]      = atr / close
        feat["vol_ratio"]    = Indicators.volume_ratio(df)
        feat["obv_slope"]    = Indicators.obv(df).diff(5)

        vwap = Indicators.vwap(df)
        feat["price_vs_vwap"] = (close - vwap) / close

        adx, plus_di, minus_di = Indicators.adx(df)
        feat["adx"]           = adx
        feat["di_diff"]       = plus_di - minus_di

        feat["body_pct"]  = (close - df["Open"]).abs() / (df["High"] - df["Low"]).replace(0, np.nan)
        feat["hl_range"]  = (df["High"] - df["Low"]) / close
        feat["ret_1"]     = close.pct_change(1)
        feat["ret_5"]     = close.pct_change(5)
        feat["ret_10"]    = close.pct_change(10)

        return feat.fillna(0)
