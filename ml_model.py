"""
ml_model.py — Machine-Learning signal layer
Uses a Random Forest + Gradient Boosting ensemble trained on labelled historical
candle data. Labels are auto-generated: a candle is 'BUY' if price rises ≥ TARGET
within the next 5 candles without hitting STOP_LOSS first, 'SELL' vice-versa,
and 'HOLD' otherwise.
"""

import os
import pickle
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import config as cfg
from indicators import Indicators

log = logging.getLogger(__name__)

# ── Try importing sklearn; degrade gracefully ──────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score
    import warnings
    warnings.filterwarnings("ignore")
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    log.warning("scikit-learn not installed → ML layer disabled. "
                "Run: pip install scikit-learn")


# ─────────────────────────────────────────────────────────────────────────────
def _label_candles(
    df: pd.DataFrame,
    fwd_candles: int = 5,
    target_pct: float = None,
    sl_pct: float = None,
) -> pd.Series:
    """
    For each candle, look forward `fwd_candles` bars.
    - If price hits target first  → label 1  (BUY)
    - If price hits stop-loss first → label -1 (SELL)
    - Otherwise                   → label 0  (HOLD)
    """
    target_pct = target_pct or cfg.TARGET_PCT / 100
    sl_pct     = sl_pct     or cfg.STOP_LOSS_PCT / 100

    labels = []
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values

    for i in range(len(df) - fwd_candles):
        entry  = closes[i]
        target = entry * (1 + target_pct)
        sl     = entry * (1 - sl_pct)
        label  = 0
        for j in range(1, fwd_candles + 1):
            if highs[i+j] >= target:
                label = 1
                break
            if lows[i+j] <= sl:
                label = -1
                break
        labels.append(label)

    # Pad tail with HOLD
    labels += [0] * fwd_candles
    return pd.Series(labels, index=df.index, name="label")


# ─────────────────────────────────────────────────────────────────────────────
class MLSignalModel:
    """
    Ensemble of:
      • RandomForest (captures non-linear patterns)
      • GradientBoosting (sequential error correction)
      • Logistic Regression (linear baseline, well-calibrated probabilities)

    Final signal is a soft-vote probability average across the three.
    """

    CLASS_MAP = {-1: "SELL", 0: "HOLD", 1: "BUY"}

    def __init__(self):
        self.models: dict = {}
        self.scaler       = None
        self.trained      = False
        self.last_trained : Optional[datetime] = None
        self.feature_cols : list[str] = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────
    def _load(self):
        if not SKLEARN_AVAILABLE:
            return
        if os.path.exists(cfg.ML_MODEL_FILE):
            try:
                with open(cfg.ML_MODEL_FILE, "rb") as f:
                    bundle = pickle.load(f)
                self.models       = bundle["models"]
                self.scaler       = bundle["scaler"]
                self.feature_cols = bundle["feature_cols"]
                self.last_trained = bundle.get("last_trained")
                self.trained      = True
                log.info(f"[ML] Model loaded from {cfg.ML_MODEL_FILE} "
                         f"(trained {self.last_trained})")
            except Exception as e:
                log.warning(f"[ML] Could not load model: {e}")

    def _save(self):
        with open(cfg.ML_MODEL_FILE, "wb") as f:
            pickle.dump({
                "models": self.models,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "last_trained": self.last_trained,
            }, f)
        log.info(f"[ML] Model saved to {cfg.ML_MODEL_FILE}")

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame, symbol: str = ""):
        """Train on one stock's OHLCV history."""
        if not SKLEARN_AVAILABLE:
            return

        if len(df) < 150:
            log.warning(f"[ML] {symbol}: not enough data ({len(df)} rows) to train")
            return

        feats  = Indicators.feature_vector(df)
        labels = _label_candles(df)
        joined = feats.join(labels).dropna()
        joined = joined[joined["label"] != 0]   # drop HOLD for cleaner boundary

        if len(joined) < 60:
            log.warning(f"[ML] {symbol}: too few labelled rows after filter")
            return

        X = joined.drop(columns=["label"]).values
        y = joined["label"].values
        self.feature_cols = list(joined.drop(columns=["label"]).columns)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.models = {
            "rf": CalibratedClassifierCV(
                RandomForestClassifier(n_estimators=200, max_depth=8,
                                       class_weight="balanced", random_state=42),
                method="sigmoid", cv=3,
            ),
            "gb": GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            ),
            "lr": Pipeline([
                ("scale", StandardScaler()),
                ("clf",   LogisticRegression(max_iter=500, class_weight="balanced")),
            ]),
        }

        scores = {}
        for name, model in self.models.items():
            model.fit(X_scaled if name != "lr" else X, y)
            cv = cross_val_score(model,
                                 X_scaled if name != "lr" else X,
                                 y, cv=3, scoring="f1_macro")
            scores[name] = cv.mean()

        log.info(f"[ML] {symbol} CV F1 → " +
                 " | ".join(f"{k}:{v:.3f}" for k, v in scores.items()))
        self.trained      = True
        self.last_trained = datetime.now()
        self._save()

    def needs_retrain(self) -> bool:
        if not self.trained or self.last_trained is None:
            return True
        return (datetime.now() - self.last_trained).days >= cfg.ML_RETRAIN_DAYS

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Returns (action, confidence) where action ∈ {BUY, SELL, HOLD}
        and confidence ∈ [0, 1].
        """
        if not SKLEARN_AVAILABLE or not self.trained:
            return "HOLD", 0.0

        try:
            feats = Indicators.feature_vector(df)
            if not self.feature_cols:
                return "HOLD", 0.0

            # Align columns
            for col in self.feature_cols:
                if col not in feats.columns:
                    feats[col] = 0.0
            feats = feats[self.feature_cols]
            row   = feats.iloc[[-1]].fillna(0).values

            X_scaled = self.scaler.transform(row)

            # Collect per-class probabilities from each model
            # Classes order: [-1, 0, 1] or [0, 1] depending on training data
            all_probas = []
            for name, model in self.models.items():
                x = X_scaled if name != "lr" else row
                proba = model.predict_proba(x)[0]
                classes = model.classes_
                p_dict = dict(zip(classes, proba))
                all_probas.append({
                    "buy":  p_dict.get(1,  0.0),
                    "sell": p_dict.get(-1, 0.0),
                    "hold": p_dict.get(0,  0.0),
                })

            avg_buy  = np.mean([p["buy"]  for p in all_probas])
            avg_sell = np.mean([p["sell"] for p in all_probas])
            avg_hold = np.mean([p["hold"] for p in all_probas])

            if avg_buy >= avg_sell and avg_buy >= avg_hold:
                return "BUY",  avg_buy
            elif avg_sell >= avg_buy and avg_sell >= avg_hold:
                return "SELL", avg_sell
            else:
                return "HOLD", avg_hold

        except Exception as e:
            log.warning(f"[ML] Predict error: {e}")
            return "HOLD", 0.0
