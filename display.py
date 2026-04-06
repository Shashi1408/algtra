"""
display.py — Rich CLI output for the trading terminal
No web UI required; everything renders in the terminal.
"""

import os
from datetime import datetime
from colorama import Fore, Back, Style, init
from tabulate import tabulate

from engine import Signal, Position

init(autoreset=True)

BOLD  = "\033[1m"
RESET = Style.RESET_ALL

# ─────────────────────────────────────────────────────────────────────────────
def clear():
    os.system("cls" if os.name == "nt" else "clear")


def banner(mode: str, capital: float, realised: float, unrealised: float, daily_loss: float):
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    mode_color = Fore.YELLOW if "DRY" in mode.upper() else Fore.GREEN
    pnl_color  = Fore.GREEN if realised >= 0 else Fore.RED
    upnl_color = Fore.GREEN if unrealised >= 0 else Fore.RED

    print(f"\n{Fore.CYAN}{'═'*78}")
    print(f"  {BOLD}⚡ ZERODHA INTRADAY TRADING SYSTEM{RESET}  "
          f"│  {Fore.WHITE}{now}")
    print(f"  Mode: {mode_color}{BOLD}{mode}{RESET}"
          f"  │  Capital: ₹{capital:,.0f}"
          f"  │  Realised P&L: {pnl_color}₹{realised:+,.2f}{RESET}"
          f"  │  Unrealised: {upnl_color}₹{unrealised:+,.2f}{RESET}"
          f"  │  Daily Loss: {Fore.RED}₹{daily_loss:,.2f}{RESET}")
    print(f"{Fore.CYAN}{'═'*78}{RESET}\n")


def section(title: str):
    print(f"\n{Fore.CYAN}{BOLD}  ── {title} {'─'*(60-len(title))}{RESET}")


def print_signals(signals: list):
    if not signals:
        print(f"  {Fore.YELLOW}No signals generated this tick.{RESET}")
        return

    rows = []
    for s in signals:
        if s.action == "BUY":
            ac = f"{Fore.GREEN}{BOLD}▲ BUY {RESET}"
        elif s.action == "SELL":
            ac = f"{Fore.RED}{BOLD}▼ SELL{RESET}"
        else:
            ac = f"{Fore.YELLOW}  HOLD{RESET}"

        conf_str = f"{s.confidence:.0%}" if s.confidence else "—"
        rows.append([
            f"{Fore.WHITE}{s.symbol}{RESET}",
            ac,
            f"₹{s.price:>9.2f}",
            f"{s.score}",
            conf_str,
            f"₹{s.stop_loss:.2f}" if s.stop_loss else "—",
            f"₹{s.target:.2f}"    if s.target    else "—",
            str(s.qty)            if s.qty        else "—",
            s.timestamp,
        ])

    print(tabulate(
        rows,
        headers=["Symbol", "Action", "Price", "Score", "ML Conf",
                 "Stop-Loss", "Target", "Qty", "Time"],
        tablefmt="rounded_outline",
    ))

    # Reason lines for actionable signals
    for s in signals:
        if s.action != "HOLD" and s.reasons:
            color = Fore.GREEN if s.action == "BUY" else Fore.RED
            print(f"  {color}{s.symbol}{RESET}  →  " +
                  "  |  ".join(s.reasons[:4]))


def print_positions(positions: dict, current_prices: dict):
    if not positions:
        print(f"  {Fore.YELLOW}No open positions.{RESET}")
        return

    rows = []
    for sym, pos in positions.items():
        cp  = current_prices.get(sym, pos.entry)
        pnl = ((cp - pos.entry) * pos.qty if pos.side == "LONG"
                else (pos.entry - cp) * pos.qty)
        pnl_c = Fore.GREEN if pnl >= 0 else Fore.RED
        rows.append([
            f"{Fore.WHITE}{sym}{RESET}",
            f"{Fore.GREEN}LONG{RESET}" if pos.side == "LONG" else f"{Fore.RED}SHORT{RESET}",
            f"₹{pos.entry:.2f}",
            f"₹{cp:.2f}",
            f"₹{pos.stop_loss:.2f}",
            f"₹{pos.target:.2f}",
            str(pos.qty),
            f"{pnl_c}₹{pnl:+.2f}{RESET}",
            pos.entry_time,
        ])

    print(tabulate(
        rows,
        headers=["Symbol", "Side", "Entry", "CMP", "SL", "TP", "Qty", "P&L", "Since"],
        tablefmt="rounded_outline",
    ))


def print_trade_log(trade_log: list):
    if not trade_log:
        print(f"  {Fore.YELLOW}No closed trades yet.{RESET}")
        return

    rows = []
    for t in trade_log[-10:]:   # last 10
        pnl_c = Fore.GREEN if t["pnl"] >= 0 else Fore.RED
        rows.append([
            t["time"], t["symbol"],
            f"{Fore.GREEN}LONG{Style.RESET_ALL}" if t["side"] == "LONG" else f"{Fore.RED}SHORT{Style.RESET_ALL}",
            f"₹{t['entry']:.2f}", f"₹{t['exit']:.2f}",
            t["qty"],
            f"{pnl_c}₹{t['pnl']:+.2f}{Style.RESET_ALL}",
            t["reason"],
        ])
    print(tabulate(
        rows,
        headers=["Time", "Symbol", "Side", "Entry", "Exit", "Qty", "P&L", "Reason"],
        tablefmt="rounded_outline",
    ))


def print_screen_results(candidates: list[str]):
    if not candidates:
        print(f"  {Fore.YELLOW}No candidates from screener.{RESET}")
        return
    print(f"  {Fore.CYAN}Screened candidates:{RESET}  " +
          "  ".join(f"{Fore.WHITE}{s}{RESET}" for s in candidates))


def shutdown_summary(trade_log: list, realised: float):
    print(f"\n{Fore.CYAN}{'═'*78}")
    print(f"  {BOLD}SESSION COMPLETE{RESET}")
    print(f"{'═'*78}\n")
    print(f"  Total Realised P&L: "
          f"{'%s' % Fore.GREEN if realised >= 0 else Fore.RED}"
          f"₹{realised:+,.2f}{RESET}")
    print(f"  Total Trades: {len(trade_log)}\n")
    if trade_log:
        wins  = [t for t in trade_log if t["pnl"] > 0]
        loss  = [t for t in trade_log if t["pnl"] <= 0]
        print(f"  Win rate: {len(wins)}/{len(trade_log)} = "
              f"{len(wins)/len(trade_log)*100:.1f}%")
        if wins:
            print(f"  Avg win:  ₹{sum(t['pnl'] for t in wins)/len(wins):.2f}")
        if loss:
            print(f"  Avg loss: ₹{sum(t['pnl'] for t in loss)/len(loss):.2f}")
        print()
        print_trade_log(trade_log)
