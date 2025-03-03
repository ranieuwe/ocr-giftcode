import os
import requests
import time
import hashlib
import json
import csv
import argparse
import sys
from datetime import datetime

# Configuration
LOGIN_URL = "https://wos-giftcode-api.centurygame.com/api/player"
REDEEM_URL = "https://wos-giftcode-api.centurygame.com/api/gift_code"
WOS_ENCRYPT_KEY = "tB87#kPtkxqOS2"  # The secret key

DELAY = 1 # Seconds between each redemption, less than 1s may result in being blocked
RETRY_DELAY = 2  # Seconds between retries
MAX_RETRIES = 3  # Max retry attempts per request

script_dir = os.path.dirname(os.path.abspath(__file__)) # store log in same directory as script
LOG_FILE = os.path.join(script_dir, "redeemed_codes.txt")

RESULT_MESSAGES = {
    "SUCCESS": "Successfully redeemed",
    "RECEIVED": "Already redeemed",
    "TIME ERROR": "Code has expired",
    "TIMEOUT RETRY": "Server requested retry",
}

counters = {
    "success": 0,
    "already_redeemed": 0,
    "errors": 0,
}

# Log messages to file and console
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {message}"

    try:
        print(log_entry)

    except UnicodeEncodeError:
        cleaned = log_entry.encode('utf-8', errors='replace').decode('ascii', errors='replace')
        print(cleaned)
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

# Generate the sign, an MD5 hash sent with the POST payload
def encode_data(data):
    secret = WOS_ENCRYPT_KEY
    sorted_keys = sorted(data.keys())

    encoded_data = "&".join(
        [
            f"{key}={json.dumps(data[key]) if isinstance(data[key], dict) else data[key]}"
            for key in sorted_keys
        ]
    )

    return {"sign": hashlib.md5(f"{encoded_data}{secret}".encode()).hexdigest(), **data}

# Send POST and handle retries if failed
def make_request(url, payload):
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("msg", "").strip('.') == "TIMEOUT RETRY":
                    if attempt < MAX_RETRIES - 1:
                        log(f"Attempt {attempt+1}: Server requested retry")
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        return response
                
                return response
            
            log(f"Attempt {attempt+1} failed: HTTP {response.status_code}")
        
        except requests.exceptions.RequestException as e:
            log(f"Attempt {attempt+1} failed: {str(e)}")
        
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
    
    return None

# Redeem a gift code for a player and return the response
def redeem_gift_code(fid, cdk):
    try:
        login_payload = encode_data({"fid": fid, "time": int(time.time() * 1000)})
        login_resp = make_request(LOGIN_URL, login_payload)
        
        if not login_resp or login_resp.json().get("code") != 0:
            return {"msg": "Login failed"}

        nickname = login_resp.json().get("data", {}).get("nickname")
        log(f"Processing {nickname or 'Unknown Player'} ({fid})")

        redeem_payload = encode_data({
            "fid": fid,
            "cdk": cdk,
            "time": int(time.time() * 1000)
        })

        redeem_resp = make_request(REDEEM_URL, redeem_payload)
        return redeem_resp.json() if redeem_resp else {"msg": "Redemption failed"}
    
    except Exception as e:
        return {"msg": f"Error: {str(e)}"}

# Read player IDs from a CSV file
def read_player_ids_from_csv(file_path):
    player_ids = []
    with open(file_path, mode="r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                player_ids.append(row[0])
    return player_ids

# Print summary of actions
def print_summary():
    log("\n=== Redemption Complete ===")
    log(f"Successfully redeemed: {counters['success']}")
    log(f"Already redeemed: {counters['already_redeemed']}")
    log(f"Errors: {counters['errors']}")

# Main script
if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Redeem gift codes for player IDs from a CSV file.")
    parser.add_argument("--csv", required=True, help="Path to the CSV file containing player IDs.")
    parser.add_argument("--code", required=True, help="The gift code to redeem.")
    args = parser.parse_args()

    # Log initialization message
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"\n=== Starting redemption for gift code: {args.code} at {start_time} ===")

    # Read player IDs from CSV file
    try:
        with open(args.csv, "r") as f:
            player_ids = [row[0].strip() for row in csv.reader(f) if row]
        log(f"Loaded {len(player_ids)} player IDs from {args.csv}")
    except FileNotFoundError:
        log(f"Error: CSV file '{args.csv}' not found")
        sys.exit(1)
    except Exception as e:
        log(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Validate the gift code
    if not args.code:
        log("Error: Gift code is required.")
        sys.exit(1)

    # Redeem gift code for each player, update counters
    for fid in player_ids:
        result = redeem_gift_code(fid, args.code)

        raw_msg = result.get('msg', 'Unknown error').strip('.')
        friendly_msg = RESULT_MESSAGES.get(raw_msg, raw_msg)

        if raw_msg == "SUCCESS":
            counters["success"] += 1
        elif raw_msg == "RECEIVED":
            counters["already_redeemed"] += 1
        elif raw_msg == "TIME ERROR":
            log("Code has expired! Script will now exit.")
            print_summary()
            sys.exit(1)
        else:
            counters["errors"] += 1

        log(f"Result: {friendly_msg}")
        time.sleep(DELAY)

    print_summary()