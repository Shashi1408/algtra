"""
main.py — Zerodha Intraday Trading Bot
=======================================
Fully automated: screens NSE stocks, applies 8 rule-based strategies + ML
ensemble, and executes live orders through Zerodha Kite Connect API.

Usage:
    python main.py [--paper]      # --paper forces dry-run regardless of credentials

Setup:
    pip install kiteconnect pandas numpy scikit-learn schedule colorama tabulate

    1. Fill in config.py with your Zerodha API key/secret/user_id
    2. Run once; a browser opens for Zerodha login
    3. Paste the request_token when prompted
    4. The bot trades automatically until 15:10 IST then squares off

⚠  DISCLAIMER: This software is for educational purposes. Automated trading
   involves significant financial risk. Use paper-trading mode until you have
   thoroughly validated the system.
"""

import sys
import time
import signal
import logging
import argparse
from datetime import datetime

import pandas as pd
import schedule

import config as cfg
from broker import ZerodhaBroker
from engine import Screener, SignalEngine, PositionManager
from ml_model import MLSignalModel
import display

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL, "INFO"),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg.LOG_FILE),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class DataFetcher:
    """
    Central OHLCV cache so each module doesn't make redundant API calls.
    Falls back to yfinance when broker is in dry-run mode.
    """

    def __init__(self, broker: ZerodhaBroker, instruments_df: pd.DataFrame):
        self.broker      = broker
        self._token_map  = {}   # symbol → instrument_token
        self._cache: dict[str, pd.DataFrame] = {}

        if not instruments_df.empty:
            eq = instruments_df[instruments_df["instrument_type"] == "EQ"]
            self._token_map = dict(zip(eq["tradingsymbol"], eq["instrument_token"]))

    def fetch(self, symbol: str, interval: str = "5minute", days: int = 5) -> pd.DataFrame:
        token = self._token_map.get(symbol)
        if token and not self.broker.dry_run:
            df = self.broker.get_historical(int(token), interval, days)
            if not df.empty:
                self._cache[symbol] = df
                return df

        # Fallback: yfinance (paper mode / development)
        try:
            import yfinance as yf
            df = yf.download(f"{symbol}.NS", period="5d", interval="5m",
                             auto_adjust=True, progress=False)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.dropna(inplace=True)
            if not df.empty:
                self._cache[symbol] = df
            return df
        except Exception as e:
            log.warning(f"[DATA] yfinance fallback failed for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_many(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        result = {}
        for sym in symbols:
            df = self.fetch(sym)
            if not df.empty:
                result[sym] = df
        return result

    def get_ltp(self, symbols: list[str]) -> dict[str, float]:
        """Last traded price for each symbol."""
        if not self.broker.dry_run and symbols:
            instruments = [f"{cfg.EXCHANGE}:{s}" for s in symbols]
            quote = self.broker.get_quote(instruments)
            return {k.replace(f"{cfg.EXCHANGE}:", ""): v.get("last_price", 0.0)
                    for k, v in quote.items()}

        # Fallback: read from cache
        prices = {}
        for sym in symbols:
            df = self._cache.get(sym)
            if df is not None and not df.empty:
                prices[sym] = float(df["Close"].iloc[-1])
        return prices


# ─────────────────────────────────────────────────────────────────────────────
class TradingBot:

    def __init__(self, force_paper: bool = False):
        log.info("Initialising Zerodha Trading Bot …")

        self.broker  = ZerodhaBroker()
        if force_paper:
            self.broker.dry_run = True
            log.info("[BOT] --paper flag set → DRY-RUN mode forced")

        instr_df      = self.broker.get_instruments(cfg.EXCHANGE)
        self.fetcher  = DataFetcher(self.broker, instr_df)
        self.screener = Screener(self.broker)
        self.engine   = SignalEngine()
        self.pm       = PositionManager(self.broker)
        self.ml       = self.engine.ml

        self.candidates    : list[str]             = []
        self.data_cache    : dict[str, pd.DataFrame] = {}
        self.last_screen   : datetime               = datetime.min
        self.running       : bool                   = True

        # Graceful shutdown on Ctrl+C / SIGTERM
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, *_):
        log.info("Shutdown signal received …")
        self.running = False

    # ── ML training ───────────────────────────────────────────────────────────
    def _maybe_retrain_ml(self):
        if not self.ml.needs_retrain():
            return
        log.info("[ML] Retraining model on current candidates …")
        for sym, df in self.data_cache.items():
            if len(df) > 150:
                self.ml.train(df, sym)
                break   # Train on one liquid stock; generalises well enough

    # ── Screening ─────────────────────────────────────────────────────────────
    def _maybe_rescreen(self):
        elapsed = (datetime.now() - self.last_screen).total_seconds() / 60
        if elapsed < cfg.SCREEN_INTERVAL_MIN:
            return

        movers          = self.broker.get_top_movers()
        self.data_cache = self.fetcher.fetch_many(movers)
        self.candidates = self.screener.run(self.data_cache)
        self.last_screen = datetime.now()

        display.section("SCREENER")
        display.print_screen_results(self.candidates)

        self._maybe_retrain_ml()

    # ── EOD squareoff ─────────────────────────────────────────────────────────
    def _squareoff_check(self):
        now_str = datetime.now().strftime("%H:%M")
        if now_str >= cfg.SQUAREOFF_TIME and self.pm.positions:
            ltps = self.fetcher.get_ltp(list(self.pm.positions.keys()))
            self.pm.squareoff_all(ltps)

    # ── Market hours guard ────────────────────────────────────────────────────
    @staticmethod
    def _is_market_open() -> bool:
        now = datetime.now()
        if now.weekday() >= 5:   # Saturday / Sunday
            return False
        t = now.hour * 60 + now.minute
        return 555 <= t <= 915   # 09:15 – 15:15 IST

    # ── Main tick ─────────────────────────────────────────────────────────────
    def tick(self):
        if not self._is_market_open():
            log.info("[BOT] Market closed. Waiting …")
            return

        self._squareoff_check()
        self._maybe_rescreen()

        if not self.candidates:
            log.info("[BOT] No candidates. Waiting for next screen …")
            return

        # Refresh data for current candidates
        new_data = self.fetcher.fetch_many(self.candidates)
        self.data_cache.update(new_data)

        # Live prices
        ltps = self.fetcher.get_ltp(self.candidates)

        signals = []
        for sym in self.candidates:
            df = self.data_cache.get(sym)
            if df is None or len(df) < 50:
                continue
            try:
                sig = self.engine.analyse(sym, df)
                signals.append(sig)

                # Exit check first
                ltp = ltps.get(sym, sig.price)
                self.pm.check_exits(sym, ltp)

                # Open if actionable and no existing position
                if sig.action in ("BUY", "SELL") and sym not in self.pm.positions:
                    self.pm.open(sig)

            except Exception as e:
                log.error(f"[BOT] Error analysing {sym}: {e}", exc_info=True)

        # ── Render CLI ────────────────────────────────────────────────────────
        display.clear()
        mode = "DRY-RUN / PAPER" if self.broker.dry_run else "LIVE TRADING"
        unreal = self.pm.unrealised_pnl(ltps)
        display.banner(mode, cfg.CAPITAL, self.pm.realised_pnl, unreal, self.pm.daily_loss)

        display.section("SIGNALS")
        display.print_signals(signals)

        display.section("OPEN POSITIONS")
        display.print_positions(self.pm.positions, ltps)

        display.section("RECENT TRADES")
        display.print_trade_log(self.pm.trade_log)

    # ── Run loop ──────────────────────────────────────────────────────────────
    def run(self):
        log.info("=" * 65)
        log.info("  ZERODHA INTRADAY TRADING SYSTEM")
        log.info(f"  Mode: {'DRY-RUN' if self.broker.dry_run else 'LIVE'}")
        log.info(f"  Capital: ₹{cfg.CAPITAL:,}")
        log.info(f"  Max positions: {cfg.MAX_OPEN_POSITIONS}")
        log.info(f"  Signal threshold: {cfg.SIGNAL_THRESHOLD}/10")
        log.info(f"  Square-off: {cfg.SQUAREOFF_TIME} IST")
        log.info("=" * 65)

        self.tick()   # immediate first tick
        schedule.every(cfg.POLL_INTERVAL_SEC).seconds.do(self.tick)

        while self.running:
            schedule.run_pending()
            time.sleep(1)

        # ── Cleanup on exit ───────────────────────────────────────────────────
        if self.pm.positions:
            log.info("Squaring off remaining positions before exit …")
            ltps = self.fetcher.get_ltp(list(self.pm.positions.keys()))
            self.pm.squareoff_all(ltps)

        display.shutdown_summary(self.pm.trade_log, self.pm.realised_pnl)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zerodha Intraday Trading Bot")
    parser.add_argument("--paper", action="store_true",
                        help="Force paper/dry-run mode (no real orders)")
    args = parser.parse_args()

    bot = TradingBot(force_paper=args.paper)
    bot.run()
