"""
generate_token.py — Run this ONCE every morning before starting the bot.
It fetches a fresh Zerodha access token and writes it directly into config.py.

Usage:
    python generate_token.py
"""

import re
import sys
import webbrowser

# ── Try importing kiteconnect ─────────────────────────────────────────────────
try:
    from kiteconnect import KiteConnect
except ImportError:
    print("ERROR: kiteconnect not installed. Run:  pip install kiteconnect")
    sys.exit(1)

import config as cfg


def update_config(access_token: str):
    """Overwrite the ZERODHA_ACCESS_TOKEN line in config.py in-place."""
    with open("config.py", "r") as f:
        content = f.read()

    new_line    = f'ZERODHA_ACCESS_TOKEN = "{access_token}"'
    updated     = re.sub(
        r'^ZERODHA_ACCESS_TOKEN\s*=\s*["\'].*?["\']',
        new_line,
        content,
        flags=re.MULTILINE,
    )

    with open("config.py", "w") as f:
        f.write(updated)

    print(f"\n  ✓ config.py updated with today's access token.")


def main():
    print("=" * 60)
    print("  ZERODHA DAILY TOKEN GENERATOR")
    print("=" * 60)

    if cfg.ZERODHA_API_KEY == "your_api_key_here":
        print("\nERROR: Please fill in ZERODHA_API_KEY in config.py first.")
        sys.exit(1)

    kite      = KiteConnect(api_key=cfg.ZERODHA_API_KEY)
    login_url = kite.login_url()

    print(f"\n  Step 1: Opening Zerodha login in your browser …")
    print(f"  URL: {login_url}\n")
    webbrowser.open(login_url)

    print("  Step 2: Log in with your Zerodha credentials + TOTP.")
    print("  Step 3: After login you'll be redirected to a URL like:")
    print("          https://127.0.0.1/?request_token=XXXXXXXX&status=success")
    print("\n  Copy the 'request_token' value from that URL.")

    request_token = input("\n  Paste request_token here: ").strip()

    if not request_token:
        print("ERROR: No token entered. Exiting.")
        sys.exit(1)

    try:
        print("\n  Generating session …")
        session      = kite.generate_session(
            request_token, api_secret=cfg.ZERODHA_API_SECRET
        )
        access_token = session["access_token"]

        print(f"  Access Token : {access_token}")
        update_config(access_token)

        # Quick profile verify
        kite.set_access_token(access_token)
        profile = kite.profile()
        print(f"  Logged in as : {profile.get('user_name')} ({profile.get('user_id')})")
        print("\n  ✓ Ready to trade! Now run:  python main.py")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: Could not generate session — {e}")
        print("Make sure your API secret is correct in config.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
