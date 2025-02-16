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

script_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(script_dir, "redeemed_codes.txt")

# Log messages to file and console
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {message}"
    print(log_entry)
    with open(LOG_FILE, "a") as f:
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
    sign = hashlib.md5(f"{encoded_data}{secret}".encode()).hexdigest()
    return {"sign": sign, **data}

# Redeem a gift code for a player and return the response
def redeem_gift_code(fid, cdk):
    try:
        # Step 1: Login (first POST request)
        login_params = {
            "fid": fid,
            "time": int(time.time() * 1000)  # Current timestamp in milliseconds
        }
        login_payload = encode_data(login_params)
        
        login_resp = requests.post(LOGIN_URL, json=login_payload)
        if login_resp.json().get("code") != 0:
            return {"status": "Failed", "message": "Login error", "response": login_resp.json()}
        
        # Step 2: Redeem Gift Code (second POST request)
        redeem_params = {
            "fid": fid,
            "cdk": cdk,
            "time": int(time.time() * 1000)
        }
        redeem_payload = encode_data(redeem_params)
        
        redeem_resp = requests.post(REDEEM_URL, json=redeem_payload)
        return redeem_resp.json()
    
    except Exception as e:
        return {"msg": f"Error: {str(e)}"}

# Read player IDs from a CSV file
def read_player_ids_from_csv(file_path):
    player_ids = []
    with open(file_path, mode="r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if row:  # Skip empty rows
                player_ids.append(row[0])  # Assuming player IDs are in the first column
    return player_ids

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

    # Redeem gift code for each player
    for fid in player_ids:
        log(f"Processing Player ID: {fid}...")
        result = redeem_gift_code(fid, args.code)
        log(f"Result: {result.get('msg', 'Unknown error')}")
        time.sleep(1)  # Avoid rate limiting