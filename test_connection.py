"""
Quick Telegram + Network Diagnostic
Run this in the SAME folder as your .env file:
    python test_connection.py
"""
import os
print("=" * 50)
print("  CONNECTION DIAGNOSTIC")
print("=" * 50)

# 1. Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] .env loaded")
except:
    print("[WARN] No dotenv — using system env")

# 2. Show what we loaded
token = os.getenv("TELEGRAM_TOKEN", "NOT SET")
chat_id = os.getenv("TELEGRAM_CHAT_ID", "NOT SET")
gemini = os.getenv("GEMINI_API_KEY", "NOT SET")

print(f"\nTELEGRAM_TOKEN = {token[:12]}...{token[-5:]}" if len(token) > 20 else f"\nTELEGRAM_TOKEN = '{token}'  *** PROBLEM ***")
print(f"TELEGRAM_CHAT_ID = '{chat_id}'")
print(f"GEMINI_API_KEY = {'SET (' + gemini[:8] + '...)' if len(gemini) > 10 else '*** NOT SET ***'}")

if token in ("NOT SET", "PASTE_YOUR_TOKEN_HERE", "your_telegram_bot_token_here"):
    print("\n*** Your Telegram token is not configured! ***")
    print("Edit your .env file and set TELEGRAM_TOKEN=<your actual token>")
    input("\nPress ENTER to close...")
    exit()

# 3. Network tests
import requests
print("\n--- Network Tests ---")

try:
    r = requests.get("https://www.google.com", timeout=10)
    print(f"[1] Google.com ........... OK ({r.status_code})")
except Exception as e:
    print(f"[1] Google.com ........... FAILED: {e}")

try:
    r = requests.get("https://api.telegram.org", timeout=10)
    print(f"[2] api.telegram.org ..... OK ({r.status_code})")
except Exception as e:
    print(f"[2] api.telegram.org ..... FAILED: {e}")
    print("    ^ Telegram is BLOCKED by firewall/antivirus/VPN")

# 4. Send test message
print("\n--- Telegram Send Test ---")
try:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    res = requests.post(url, data={
        "chat_id": chat_id,
        "text": "✅ Connection test successful!"
    }, timeout=15)
    print(f"Status: {res.status_code}")
    print(f"Response: {res.text[:500]}")
    
    if res.status_code == 200:
        print("\n✅ SUCCESS! Check your Telegram — you should see the test message.")
    elif res.status_code == 401:
        print("\n❌ BAD TOKEN — your TELEGRAM_TOKEN is invalid.")
    elif res.status_code == 400:
        print("\n❌ BAD CHAT ID — your TELEGRAM_CHAT_ID is wrong.")
    else:
        print(f"\n❌ Unexpected error: {res.status_code}")
        
except requests.exceptions.SSLError as e:
    print(f"❌ SSL ERROR: {e}")
    print("Your antivirus is intercepting HTTPS connections.")
    print("Try adding python.exe to your antivirus exceptions.")
except requests.exceptions.ConnectionError as e:
    print(f"❌ CONNECTION ERROR: {e}")
    print("Telegram is being blocked. Check firewall/antivirus/VPN.")
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}: {e}")

# 5. Check if old bot worked
print("\n--- File Check ---")
bot_file = "market_bot_v24.py"
if os.path.exists(bot_file):
    with open(bot_file, 'r') as f:
        content = f.read(2000)
    if "Network Diagnostic" in content:
        print(f"[OK] {bot_file} is the LATEST version")
    else:
        print(f"[!!] {bot_file} is the OLD version — replace it with the latest download!")
else:
    print(f"[!!] {bot_file} not found in this folder")

print(f"\nCurrent folder: {os.getcwd()}")
print(f"Files here: {os.listdir('.')}")

input("\nPress ENTER to close...")
