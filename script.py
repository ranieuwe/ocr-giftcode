import requests
import time
import hashlib
import json

# Configuration
LOGIN_URL = "https://wos-giftcode-api.centurygame.com/api/player"
REDEEM_URL = "https://wos-giftcode-api.centurygame.com/api/gift_code"
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

# Example usage
player_ids = ["157701025", "98765432"]  # Replace with your list of player IDs
gift_code = "ILoveU"  # Replace with your gift code

for fid in player_ids:
    result = redeem_gift_code(fid, gift_code)
    print(f"Player {fid}: {result['msg']}")
    time.sleep(1)  # Avoid rate limiting