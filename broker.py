"""
broker.py — Zerodha Kite Connect wrapper
Handles authentication, live quotes, historical OHLCV, and order execution.

Fallback data source (paper/dry-run mode): nsetools
  → pulls live quotes directly from NSE India website (no API key needed).
  → install: pip install nsetools
"""

import os
import logging
import webbrowser
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

import config as cfg

log = logging.getLogger(__name__)

# ── Try importing kiteconnect ─────────────────────────────────────────────────
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    log.warning("kiteconnect not installed. Run:  pip install kiteconnect")

# ── Try importing nsetools (paper-mode / dry-run fallback) ───────────────────
try:
    from nsetools import Nse as NseTools
    _nse = NseTools()
    NSE_TOOLS_AVAILABLE = True
    log.info("[BROKER] nsetools available — will use for paper-mode data")
except ImportError:
    NSE_TOOLS_AVAILABLE = False
    _nse = None
    log.warning("nsetools not installed. Run:  pip install nsetools")


# ─────────────────────────────────────────────────────────────────────────────
class BrokerError(Exception):
    pass


# ─────────────────────────────────────────────────────────────────────────────
class ZerodhaBroker:
    """
    Thin wrapper around KiteConnect.
    Falls back to a DRY-RUN / paper-trading mode when credentials are absent
    or kiteconnect is not installed, so the rest of the system keeps running.
    """

    def __init__(self):
        self.kite: Optional["KiteConnect"] = None
        self.dry_run = True
        self._connect()

    # ── Authentication ────────────────────────────────────────────────────────
    def _connect(self):
        if not KITE_AVAILABLE:
            log.warning("[BROKER] kiteconnect not available → DRY-RUN mode")
            return

        if cfg.ZERODHA_API_KEY == "your_api_key_here":
            log.warning("[BROKER] Credentials not set in config.py → DRY-RUN mode")
            return

        try:
            self.kite = KiteConnect(api_key=cfg.ZERODHA_API_KEY)

            # Priority: 1) hardcoded token in config  2) saved file  3) browser login
            token = (cfg.ZERODHA_ACCESS_TOKEN
                     if cfg.ZERODHA_ACCESS_TOKEN not in ("", "your_access_token_here")
                     else self._load_token())

            if token:
                self.kite.set_access_token(token)
                # Quick validity check
                self.kite.profile()
                self.dry_run = False
                source = "config.py" if token == getattr(cfg, "ZERODHA_ACCESS_TOKEN", "") else "saved file"
                log.info(f"[BROKER] ✓ Connected to Zerodha (token from {source})")
            else:
                self._fresh_login()
        except Exception as e:
            log.error(f"[BROKER] Connection failed: {e} → DRY-RUN mode")

    def _fresh_login(self):
        """Open browser for Kite login and exchange request_token for access_token."""
        login_url = self.kite.login_url()
        print(f"\n{'═'*65}")
        print("  ZERODHA LOGIN REQUIRED")
        print(f"{'═'*65}")
        print(f"  Opening browser → {login_url}")
        print("  After login, copy the 'request_token' from the redirect URL.")
        print(f"{'═'*65}\n")
        webbrowser.open(login_url)
        request_token = input("  Paste request_token here: ").strip()

        data = self.kite.generate_session(request_token, api_secret=cfg.ZERODHA_API_SECRET)
        access_token = data["access_token"]
        self.kite.set_access_token(access_token)
        self._save_token(access_token)
        self.dry_run = False
        log.info("[BROKER] ✓ Fresh login successful")

    def _load_token(self) -> Optional[str]:
        if os.path.exists(cfg.ACCESS_TOKEN_FILE):
            with open(cfg.ACCESS_TOKEN_FILE) as f:
                return f.read().strip() or None
        return None

    def _save_token(self, token: str):
        with open(cfg.ACCESS_TOKEN_FILE, "w") as f:
            f.write(token)

    # ── Market data ───────────────────────────────────────────────────────────
    def get_instruments(self, exchange: str = "NSE") -> pd.DataFrame:
        """Return full instrument list for an exchange."""
        if self.dry_run:
            return pd.DataFrame()
        try:
            instr = self.kite.instruments(exchange)
            return pd.DataFrame(instr)
        except Exception as e:
            log.error(f"[BROKER] get_instruments: {e}")
            return pd.DataFrame()

    def get_quote(self, instruments: list[str]) -> dict:
        """
        instruments: list of 'NSE:RELIANCE' style strings
        Returns Kite quote dict.
        """
        if self.dry_run:
            return {}
        try:
            return self.kite.quote(instruments)
        except Exception as e:
            log.error(f"[BROKER] get_quote: {e}")
            return {}

    def get_historical(
        self,
        instrument_token: int,
        interval: str = "5minute",
        days: int = 5,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles from Kite historical API."""
        if self.dry_run:
            return pd.DataFrame()
        try:
            to_date   = datetime.now()
            from_date = to_date - timedelta(days=days)
            records   = self.kite.historical_data(
                instrument_token, from_date, to_date, interval
            )
            df = pd.DataFrame(records)
            df.rename(columns={"date": "Date", "open": "Open", "high": "High",
                                "low": "Low", "close": "Close", "volume": "Volume"},
                      inplace=True)
            df.set_index("Date", inplace=True)
            return df
        except Exception as e:
            log.error(f"[BROKER] get_historical({instrument_token}): {e}")
            return pd.DataFrame()

    def get_top_movers(self) -> list[str]:
        """
        Return top gainers/high-volume symbols for the day from Kite.
        Kite doesn't have a direct 'gainers' API, so we pull all NSE
        instruments, fetch quotes in batches, and rank by % change.
        """
        if self.dry_run:
            return self._nsetools_top_movers()
        try:
            instr_df = self.get_instruments("NSE")
            # Filter equity, price range
            eq = instr_df[
                (instr_df["instrument_type"] == "EQ") &
                (instr_df["last_price"].between(cfg.MIN_PRICE, cfg.MAX_PRICE))
            ].copy()

            # Batch quote (Kite allows ~500 at once)
            symbols = [f"NSE:{s}" for s in eq["tradingsymbol"].tolist()[:500]]
            batches = [symbols[i:i+200] for i in range(0, len(symbols), 200)]
            rows = []
            for batch in batches:
                q = self.kite.quote(batch)
                for key, v in q.items():
                    ohlc = v.get("ohlc", {})
                    prev_close = ohlc.get("close", 0)
                    ltp = v.get("last_price", 0)
                    vol = v.get("volume", 0)
                    chg_pct = ((ltp - prev_close) / prev_close * 100) if prev_close else 0
                    rows.append({"symbol": key.replace("NSE:", ""),
                                 "ltp": ltp, "chg_pct": chg_pct, "volume": vol})

            df = pd.DataFrame(rows)
            # Score: combine % change and volume rank
            df["vol_rank"] = df["volume"].rank(pct=True)
            df["score"]    = df["chg_pct"].abs() * 0.6 + df["vol_rank"] * 40
            top = df.nlargest(cfg.TOP_GAINERS_N, "score")
            return top["symbol"].tolist()
        except Exception as e:
            log.error(f"[BROKER] get_top_movers: {e}")
            return []

    # ── nsetools helpers (paper / dry-run mode) ───────────────────────────────
    def _nsetools_top_movers(self) -> list[str]:
        """
        Use nsetools to fetch live top gainers + high-volume stocks directly
        from NSE India. No API key required. Used only in dry-run/paper mode.
        """
        if not NSE_TOOLS_AVAILABLE:
            log.warning("[BROKER] nsetools not available — returning default watchlist")
            return [
                "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
                "WIPRO", "AXISBANK", "SBIN", "TATAMOTORS", "BAJFINANCE",
                "HINDUNILVR", "KOTAKBANK", "LT", "MARUTI", "SUNPHARMA",
                "ULTRACEMCO", "TITAN", "NESTLEIND", "POWERGRID", "NTPC",
            ]
        try:
            symbols: set[str] = set()

            # 1. Top gainers from NSE
            gainers = _nse.get_top_gainers() or []
            for g in gainers[:15]:
                sym = g.get("symbol", "")
                if sym:
                    symbols.add(sym.upper())

            # 2. 52-week highs (momentum)
            highs = _nse.get_52_week_high() or []
            for h in highs[:10]:
                sym = h.get("symbol", "")
                if sym:
                    symbols.add(sym.upper())

            # 3. Stocks in high-liquidity indices
            for index in ["NIFTY 50", "NIFTY BANK", "NIFTY IT"]:
                try:
                    idx_stocks = _nse.get_stocks_in_index(index) or []
                    for s in idx_stocks:
                        symbols.add(s.upper())
                except Exception:
                    pass

            # Filter by price range using live quotes
            filtered = []
            for sym in list(symbols):
                try:
                    q = _nse.get_quote(sym.lower())
                    if q:
                        ltp = float(q.get("lastPrice") or q.get("last_price") or 0)
                        if cfg.MIN_PRICE <= ltp <= cfg.MAX_PRICE:
                            filtered.append(sym)
                except Exception:
                    continue

            result = filtered[:cfg.TOP_GAINERS_N] if filtered else list(symbols)[:cfg.TOP_GAINERS_N]
            log.info(f"[BROKER] nsetools returned {len(result)} movers")
            return result

        except Exception as e:
            log.error(f"[BROKER] nsetools top movers failed: {e}")
            return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
                    "SBIN", "AXISBANK", "WIPRO", "TATAMOTORS", "BAJFINANCE"]

    def nsetools_get_ltp(self, symbol: str) -> float:
        """Fetch last traded price via nsetools (dry-run mode)."""
        if not NSE_TOOLS_AVAILABLE:
            return 0.0
        try:
            q = _nse.get_quote(symbol.lower())
            if q:
                return float(q.get("lastPrice") or q.get("last_price") or 0)
        except Exception as e:
            log.warning(f"[BROKER] nsetools LTP failed for {symbol}: {e}")
        return 0.0

    # ── Order execution ───────────────────────────────────────────────────────
    def place_order(
        self,
        symbol: str,
        transaction_type: str,   # "BUY" | "SELL"
        quantity: int,
        order_type: str = cfg.ORDER_TYPE,
        price: float = 0.0,
        trigger_price: float = 0.0,
    ) -> Optional[str]:
        """Place an order on Zerodha. Returns order_id or None on failure."""
        kite_txn = "BUY" if transaction_type == "BUY" else "SELL"
        log.info(f"[ORDER] {'🟢' if kite_txn=='BUY' else '🔴'} "
                 f"{kite_txn} {quantity}×{symbol} @ {order_type}")

        if self.dry_run:
            fake_id = f"DRYRUN-{symbol}-{datetime.now().strftime('%H%M%S')}"
            log.info(f"[ORDER] DRY-RUN → simulated order_id={fake_id}")
            return fake_id

        try:
            params = dict(
                tradingsymbol=symbol,
                exchange=cfg.EXCHANGE,
                transaction_type=kite_txn,
                quantity=quantity,
                product=cfg.PRODUCT,
                order_type=order_type,
                variety=self.kite.VARIETY_REGULAR,
            )
            if order_type == "LIMIT":
                params["price"] = round(price, 2)
            if trigger_price:
                params["trigger_price"] = round(trigger_price, 2)
                params["order_type"]    = self.kite.ORDER_TYPE_SLM

            order_id = self.kite.place_order(**params)
            log.info(f"[ORDER] ✓ order_id={order_id}")
            return str(order_id)
        except Exception as e:
            log.error(f"[ORDER] Failed: {e}")
            return None

    def place_bracket_order(
        self,
        symbol: str,
        transaction_type: str,
        quantity: int,
        price: float,
        stop_loss_pts: float,
        target_pts: float,
    ) -> Optional[str]:
        """Place a Zerodha Bracket Order (BO) for automatic SL+TP."""
        if self.dry_run:
            fake_id = f"BO-DRYRUN-{symbol}-{datetime.now().strftime('%H%M%S')}"
            log.info(f"[BO] DRY-RUN → {fake_id}")
            return fake_id
        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_BO,
                tradingsymbol=symbol,
                exchange=cfg.EXCHANGE,
                transaction_type=transaction_type,
                quantity=quantity,
                product=self.kite.PRODUCT_BO,
                order_type=self.kite.ORDER_TYPE_LIMIT,
                price=round(price, 2),
                squareoff=round(target_pts, 2),
                stoploss=round(stop_loss_pts, 2),
                trailing_stoploss=round(price * cfg.TRAILING_SL_PCT / 100, 2),
            )
            log.info(f"[BO] ✓ Bracket order placed order_id={order_id}")
            return str(order_id)
        except Exception as e:
            log.error(f"[BO] Failed: {e}")
            return None

    def cancel_order(self, order_id: str):
        if self.dry_run:
            log.info(f"[ORDER] DRY-RUN cancel {order_id}")
            return
        try:
            self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR,
                                   order_id=order_id)
            log.info(f"[ORDER] Cancelled {order_id}")
        except Exception as e:
            log.error(f"[ORDER] Cancel failed: {e}")

    def get_positions(self) -> pd.DataFrame:
        if self.dry_run:
            return pd.DataFrame()
        try:
            pos = self.kite.positions()
            return pd.DataFrame(pos.get("day", []))
        except Exception as e:
            log.error(f"[BROKER] get_positions: {e}")
            return pd.DataFrame()

    def get_orders(self) -> pd.DataFrame:
        if self.dry_run:
            return pd.DataFrame()
        try:
            return pd.DataFrame(self.kite.orders())
        except Exception as e:
            log.error(f"[BROKER] get_orders: {e}")
            return pd.DataFrame()
