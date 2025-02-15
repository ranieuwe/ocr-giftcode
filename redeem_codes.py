import requests
import time
import hashlib
import json
import csv
import argparse
import sys

# Configuration
LOGIN_URL = "https://wos-giftcode-api.centurygame.com/api/player"
REDEEM_URL = "https://wos-giftcode-api.centurygame.com/api/gift_code"  # Updated URL
WOS_ENCRYPT_KEY = "tB87#kPtkxqOS2"  # The secret key

# Function to generate the sign
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

# Function to redeem a gift code for a player
def redeem_gift_code(fid, cdk):
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
        "time": int(time.time() * 1000)  # Current timestamp in milliseconds
    }
    redeem_payload = encode_data(redeem_params)
    
    redeem_resp = requests.post(REDEEM_URL, json=redeem_payload)
    return redeem_resp.json()

# Function to read player IDs from a CSV file
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

    # Validate the CSV file path
    try:
        player_ids = read_player_ids_from_csv(args.csv)
        print(f"Loaded {len(player_ids)} player IDs from {args.csv}.")
    except FileNotFoundError:
        print(f"Error: File '{args.csv}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Validate the gift code
    if not args.code:
        print("Error: Gift code is required.")
        sys.exit(1)

    # Redeem gift code for each player
    for fid in player_ids:
        print(f"Processing Player ID: {fid}...")
        result = redeem_gift_code(fid, args.code)
        print(f"Result: {result['msg']}")
        time.sleep(1)  # Avoid rate limiting