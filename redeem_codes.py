#!/usr/bin/env python3
# Whiteout Gift Code Redeemer Script Version 2.3.0
# EasyOCR Version

import os
import warnings
import requests
import time
import random
import hashlib
import json
import csv
import argparse
import sys
import base64
import easyocr
import cv2
import numpy as np
from datetime import datetime
from glob import glob

warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)

# Configuration
LOGIN_URL = "https://wos-giftcode-api.centurygame.com/api/player"
CAPTCHA_URL = "https://wos-giftcode-api.centurygame.com/api/captcha"
REDEEM_URL = "https://wos-giftcode-api.centurygame.com/api/gift_code"
WOS_ENCRYPT_KEY = "tB87#kPtkxqOS2"

DELAY = 1
RETRY_DELAY = 2
MAX_RETRIES = 3
CAPTCHA_RETRIES = 20
CAPTCHA_SLEEP = 60

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Redeem gift codes with optional OCR settings and code sources")
    parser.add_argument('--all-images', action='store_true', help='Save all captcha images regardless of OCR success')
    parser.add_argument('--use-gpu', action='store_true', help='Enable GPU for EasyOCR reader')
    parser.add_argument('--csv', type=str, help='Path to CSV file containing codes')
    parser.add_argument('--code', type=str, help='Single code to redeem')
    parser.add_argument('codes', nargs='*', help='Gift codes to redeem directly')
    return parser.parse_args()

args = parse_args()

codes = []
if args.csv:
    try:
        with open(args.csv, newline='', encoding='utf-8-sig') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                codes.extend([c.strip() for c in row if c.strip()])
    except Exception as e:
        print(f"Error reading CSV file {args.csv}: {e}")
if args.code:
    codes.append(args.code)
codes.extend(args.codes)
args.codes = codes

script_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(script_dir, "redeemed_codes.txt")
FAILED_CAPTCHA_DIR = os.path.join(script_dir, "failed_captchas")

try:
    os.makedirs(FAILED_CAPTCHA_DIR, exist_ok=True)
    rel = os.path.relpath(FAILED_CAPTCHA_DIR, script_dir)
    print(f"Created directory for captcha images: {rel}")
except Exception as e:
    print(f"Error creating directory: {str(e)}")

# Initialize OCR reader based on args.use_gpu
args.all_images = getattr(args, 'all_images', False)
args.use_gpu = getattr(args, 'use_gpu', False)
reader = easyocr.Reader(['en'], gpu=args.use_gpu)

RESULT_MESSAGES = {
    "SUCCESS": "Successfully redeemed",
    "RECEIVED": "Already redeemed",
    "SAME TYPE EXCHANGE": "Successfully redeemed (same type)",
    "TIME ERROR": "Code has expired",
    "TIMEOUT RETRY": "Server requested retry",
    "USED": "Claim limit reached, unable to claim",
}

counters = {"success": 0, "already_redeemed": 0, "errors": 0}

def save_captcha_image(img_np, fid, attempt, captcha_code):
    try:
        timestamp = int(time.time())
        image_filename = f"fid{fid}_try{attempt}_OCR_{captcha_code}_{timestamp}.png"
        full_path = os.path.join(FAILED_CAPTCHA_DIR, image_filename)
        if cv2.imwrite(full_path, img_np):
            rel_path = os.path.relpath(full_path, script_dir)
            print(f"Saved captcha image: {rel_path}")
        else:
            rel_path = os.path.relpath(full_path, script_dir)
            print(f"Failed to save captcha image: {rel_path}")
        return image_filename
    except Exception as e:
        print(f"Exception during saving captcha image: {str(e)}")
        return None

def fetch_captcha_code(fid):
    attempts = 0
    while attempts < 10:
        payload = encode_data({"fid": fid, "time": int(time.time() * 1000), "init": "0"})
        response = make_request(CAPTCHA_URL, payload)
        if response and response.status_code == 200:
            try:
                captcha_data = response.json()
                if captcha_data.get("code") == 1 and captcha_data.get("msg") == "CAPTCHA GET TOO FREQUENT.":
                    log("Captcha fetch too frequent, sleeping before retry...")
                    time.sleep(CAPTCHA_SLEEP)
                    continue

                if "data" in captcha_data and "img" in captcha_data["data"]:
                    img_field = captcha_data["data"]["img"]
                    if img_field.startswith("data:image"):
                        img_base64 = img_field.split(",", 1)[1]
                    else:
                        img_base64 = captcha_data["data"]["img"]
                    img_bytes = base64.b64decode(img_base64)

                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    result = reader.readtext(img_bytes, detail=0)
                    captcha_code = result[0].strip().replace(' ', '') if result else 'EMPTY'

                    if getattr(args, 'all_images', False) or not (captcha_code.isalnum() and len(captcha_code) == 4):
                        save_captcha_image(img_np, fid, attempts, captcha_code)

                    if captcha_code.isalnum() and len(captcha_code) == 4:
                        log(f"Recognized captcha: {captcha_code}")
                        return captcha_code
                    else:
                        log(f"Invalid captcha format: '{captcha_code}', refetching...")
                else:
                    log("Captcha image missing in response, refetching...")
            except Exception as e:
                failed_path = os.path.join(FAILED_CAPTCHA_DIR, f"fid{fid}_exception_{int(time.time())}.png")
                with open(failed_path, "wb") as f:
                    f.write(img_bytes)
                rel_failed = os.path.relpath(failed_path, script_dir)
                log(f"Saved failed captcha image to {rel_failed}")
                log(f"Error solving captcha: {str(e)}")
        else:
            log("Failed to fetch captcha, retrying...")

        attempts += 1
        time.sleep(random.uniform(2.0, 5.0))

    raise Exception("Failed to fetch valid captcha after multiple attempts")

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {message}"
    try:
        print(log_entry)
    except UnicodeEncodeError:
        cleaned = log_entry.encode('utf-8', errors='replace').decode('ascii', errors='replace')
        print(cleaned)
    try:
        with open(LOG_FILE, "a", encoding="utf-8-sig") as f:
            f.write(log_entry + "\n")
    except Exception as e:
        print(f"{timestamp} - LOGGING ERROR: {str(e)}")

def encode_data(data):
    secret = WOS_ENCRYPT_KEY
    sorted_keys = sorted(data.keys())
    encoded_data = "&".join([f"{key}={json.dumps(data[key]) if isinstance(data[key], dict) else data[key]}" for key in sorted_keys])
    return {"sign": hashlib.md5(f"{encoded_data}{secret}".encode()).hexdigest(), **data}

def make_request(url, payload, headers=None):
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, data=payload, headers=headers)
            if response.status_code == 200:
                return response
            log(f"Attempt {attempt+1} failed: HTTP {response.status_code}, Response: {response.text[:200]}")
        except requests.exceptions.RequestException as e:
            log(f"Attempt {attempt+1} failed: {str(e)}")
        time.sleep(RETRY_DELAY)
    return None

def redeem_gift_code(fid, cdk):
    if not str(fid).strip().isdigit():
        log(f"Skipping invalid FID: '{fid}'")
        return {"msg": "Invalid FID format"}
    fid = str(fid).strip()

    try:
        login_payload = encode_data({"fid": fid, "time": int(time.time() * 1000)})
        login_resp = make_request(LOGIN_URL, login_payload)
        if not login_resp:
            return {"msg": "Login request failed after retries"}

        login_data = login_resp.json()
        if login_data.get("code") != 0:
            login_msg = login_data.get('msg', 'Unknown login error')
            log(f"Login failed for {fid}: {login_data.get('code')}, {login_msg}")
            return {"msg": f"Login failed: {login_msg}"}

        nickname = login_data.get("data", {}).get("nickname")
        log(f"Processing {nickname or 'Unknown Player'} ({fid})")

        for attempt in range(CAPTCHA_RETRIES):
            captcha_code = fetch_captcha_code(fid)
            redeem_payload = encode_data({
                "fid": fid,
                "cdk": cdk,
                "captcha_code": captcha_code,
                "time": int(time.time() * 1000)
            })
            redeem_resp = make_request(REDEEM_URL, redeem_payload)
            if not redeem_resp:
                return {"msg": "Redemption request failed after retries"}

            redeem_data = redeem_resp.json()
            msg = redeem_data.get('msg', 'Unknown error').strip('.')

            if msg in ["CAPTCHA CHECK ERROR", "Sign Error", "Server requested retry"]:
                log(f"{msg}, retrying in a bit... (Attempt {attempt+1}/{CAPTCHA_RETRIES})")
                time.sleep(random.uniform(2.5, 6.5))
                continue
            elif msg == "CAPTCHA CHECK TOO FREQUENT":
                log(f"Captcha check too frequent, taking a nap for {CAPTCHA_SLEEP} seconds...")
                time.sleep(CAPTCHA_SLEEP)
                continue
            elif msg == "NOT LOGIN":
                log(f"Session expired or invalid after captcha. Skipping {fid}.")
                return {"msg": "Session expired after captcha"}
            else:
                return redeem_data

        return {"msg": "CAPTCHA retries exhausted"}

    except Exception as e:
        log(f"Unexpected error during redemption for {fid}: {str(e)}")
        return {"msg": f"Unexpected Error: {str(e)}"}

def read_player_ids_from_csv(file_path):
    """
    Reads player IDs from a CSV file.
    Handles files where IDs are one per line, OR comma-separated on one or more lines.
    Strips whitespace from each ID and ignores empty entries.
    """
    player_ids = []
    format_detected = "newline"
    try:
        # Using utf-8-sig to handle potential BOM (Byte Order Mark)
        with open(file_path, mode="r", newline="", encoding="utf-8-sig") as file:
            sample = "".join(file.readline() for _ in range(5))
            if ',' in sample:
                format_detected = "comma-separated"
            file.seek(0)

            log(f"Reading {file_path} (detected format: {format_detected})")
            reader = csv.reader(file)
            for row_num, row in enumerate(reader, 1):
                for item in row:
                    fid = item.strip()
                    if fid:
                        player_ids.append(fid)
                    elif item and not fid:
                        log(f"Warning: Ignoring whitespace-only entry in {file_path} on row {row_num}")
                if not row and format_detected == "newline":
                     log(f"Warning: Ignoring empty line in {file_path} on row {row_num}")

    except FileNotFoundError:
        raise
    except Exception as e:
        log(f"Error reading or processing CSV file {file_path}: {str(e)}")
        return []

    return player_ids

def print_summary():
    log("\n=== Redemption Complete ===")
    log(f"Successfully redeemed: {counters['success']}")
    log(f"Already redeemed: {counters['already_redeemed']}")
    log(f"Errors/Failures: {counters['errors']}")

if __name__ == "__main__":
    # Log initialization message
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"\n=== Starting redemption for gift code: {args.code} at {start_time} ===")

    # Handle *.csv input
    if args.csv == "*.csv":
        # Use the script's directory if no folder is specified
        csv_files = glob(os.path.join(script_dir, "*.csv"))
    else:
        # Use the specified folder or file
        if os.path.isdir(args.csv):
            csv_files = glob(os.path.join(args.csv, "*.csv"))
        else:
            csv_files = [args.csv]

    if not csv_files:
        log("Error: No CSV files found.")
        sys.exit(1)

    # Process all CSV files
    for csv_file in csv_files:
        try:
            player_ids = read_player_ids_from_csv(csv_file)
            log(f"Loaded {len(player_ids)} player IDs from {csv_file}")

            # Redeem gift code for each player
            for fid in player_ids:
                result = redeem_gift_code(fid, args.code)

                raw_msg = result.get('msg', 'Unknown error').strip('.')
                friendly_msg = RESULT_MESSAGES.get(raw_msg, raw_msg)

                # Exit immediately if code is expired or claim limit reached
                if raw_msg == "TIME ERROR":
                    log("Code has expired! Script will now exit.")
                    print_summary()
                    sys.exit(1)
                elif raw_msg == "USED":
                    log("Claim limit reached! Script will now exit.")
                    print_summary()
                    sys.exit(1)

                # Update counters based on result
                if raw_msg in ["SUCCESS", "SAME TYPE EXCHANGE"]:
                    counters["success"] += 1
                elif raw_msg == "RECEIVED":
                    counters["already_redeemed"] += 1
                elif raw_msg == "TIMEOUT RETRY":
                    pass
                else:
                    counters["errors"] += 1

                log(f"Result: {friendly_msg}")
                time.sleep(DELAY)

        except FileNotFoundError:
            log(f"Error: CSV file '{csv_file}' not found")
        except Exception as e:
            log(f"Error processing {csv_file}: {str(e)}")

    # Print final summary
    print_summary()