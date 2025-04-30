#!/usr/bin/env python3
# Whiteout Gift Code Redeemer Script Version 2.4.5
# Patch to enable dGPU support

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
from datetime import datetime, timedelta
from glob import glob
from PIL import Image, ImageEnhance, ImageFilter

warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)

# Configuration
LOGIN_URL = "https://wos-giftcode-api.centurygame.com/api/player"
CAPTCHA_URL = "https://wos-giftcode-api.centurygame.com/api/captcha"
REDEEM_URL = "https://wos-giftcode-api.centurygame.com/api/gift_code"
WOS_ENCRYPT_KEY = "tB87#kPtkxqOS2"

DELAY = 1
RETRY_DELAY = 2
MAX_RETRIES = 3
CAPTCHA_RETRIES = 4  # Maximum captcha retries before moving to next FID
CAPTCHA_SLEEP = 60   # Time to wait before retrying a FID that hit rate limits
MAX_CAPTCHA_ATTEMPTS = 4  # Global maximum attempts per captcha fetch
MIN_CONFIDENCE = 0.4  # Minimum confidence to accept a result

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Redeem gift codes with optional OCR settings and code sources")
    parser.add_argument('--all-images', action='store_true', help='Save all captcha images regardless of OCR success')
    parser.add_argument('--use-gpu', type=int, nargs='?', const=0, default=None, 
                        help='Enable GPU and specify device ID (0 for iGPU, 1 for dGPU, etc. Default: None for CPU)')
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

# Initialize OCR reader based on args
args.all_images = getattr(args, 'all_images', False)
args.use_gpu = getattr(args, 'use_gpu', None)

# EasyOCR GPU handling
if args.use_gpu is not None:
    import torch
    try:
        torch.cuda.set_device(args.use_gpu)
        gpu_name = torch.cuda.get_device_name(args.use_gpu)
        print(f"Using GPU device {args.use_gpu}: {gpu_name}")
        reader = easyocr.Reader(['en'], gpu=True)
    except Exception as e:
        print(f"GPU error: {str(e)}. Falling back to CPU.")
        reader = easyocr.Reader(['en'], gpu=False)
else:
    print("Using CPU only (no GPU acceleration)")
    reader = easyocr.Reader(['en'], gpu=False)

RESULT_MESSAGES = {
    "SUCCESS": "Successfully redeemed",
    "RECEIVED": "Already redeemed",
    "SAME TYPE EXCHANGE": "Successfully redeemed (same type)",
    "TIME ERROR": "Code has expired",
    "TIMEOUT RETRY": "Server requested retry",
    "USED": "Claim limit reached, unable to claim",
    "Server requested retry": "Server requested retry",
}

counters = {
    "success": 0, 
    "already_redeemed": 0, 
    "errors": 0,
    "captcha_success": 0,  # Total successful captcha decodes
    "captcha_first_try": 0,  # Captchas decoded on first attempt
    "captcha_retry": 0,  # Captchas that needed retry
    "captcha_attempts": 0,  # Total captcha attempts
    "captcha_failures": 0,  # Complete failures to decode captcha
}

error_details = {}

script_start_time = time.time()

def preprocess_captcha(image):
    """Apply multiple preprocessing techniques and return processed images"""
    # Convert PIL image to numpy array for OpenCV
    if isinstance(image, Image.Image):
        img_np = np.array(image)
        # Convert RGB to BGR (OpenCV format)
        if len(img_np.shape) > 2 and img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_np = image
        
    processed_images = []
    processed_images.append(("Original", img_np))
    
    # Method 1: Basic threshold
    if len(img_np.shape) > 2:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_np
        
    _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    processed_images.append(("Basic Threshold", thresh1))
    
    # Method 2: Adaptive threshold
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    processed_images.append(("Adaptive Threshold", adaptive_thresh))
    
    # Method 3: Otsu's thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    processed_images.append(("Otsu Threshold", otsu))
    
    # Method 4: Noise removal
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    processed_images.append(("Denoised", denoised))
    
    # Method 5: Noise removal + threshold
    _, denoised_thresh = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)
    processed_images.append(("Denoised+Threshold", denoised_thresh))
    
    # Method 6: Dilated
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=1)
    processed_images.append(("Dilated", dilated))
    
    # Method 7: Eroded
    eroded = cv2.erode(gray, kernel, iterations=1)
    processed_images.append(("Eroded", eroded))
    
    # Method 8: Edge enhancement
    edges = cv2.Canny(gray, 100, 200)
    processed_images.append(("Edges", edges))
    
    # Method 9: Morphological operations
    kernel = np.ones((1,1), np.uint8)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    processed_images.append(("Opening", opening))
    
    # Method 10: Enhanced contrast
    if isinstance(image, Image.Image):
        pil_img = image
    else:
        if len(img_np.shape) > 2 and img_np.shape[2] == 3:
            pil_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(img_np)
            
    enhanced = ImageEnhance.Contrast(pil_img).enhance(2.0)
    enhanced_np = np.array(enhanced)
    if len(enhanced_np.shape) > 2 and enhanced_np.shape[2] == 3:
        enhanced_np = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
    processed_images.append(("Enhanced Contrast", enhanced_np))
    
    # Method 11: Sharpened
    sharpened = pil_img.filter(ImageFilter.SHARPEN)
    sharpened_np = np.array(sharpened)
    if len(sharpened_np.shape) > 2 and sharpened_np.shape[2] == 3:
        sharpened_np = cv2.cvtColor(sharpened_np, cv2.COLOR_RGB2BGR)
    processed_images.append(("Sharpened", sharpened_np))
    
    # Method 12: Color filtering (for captchas with specific color text)
    if len(img_np.shape) > 2 and img_np.shape[2] == 3:
        # Extract blue channel
        blue_channel = img_np[:, :, 0]
        _, blue_thresh = cv2.threshold(blue_channel, 127, 255, cv2.THRESH_BINARY)
        processed_images.append(("Blue Channel", blue_thresh))
        
        # Create an HSV version and filter for common captcha colors
        hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
        
        # Purple-blue range
        lower_purple = np.array([100, 50, 50])
        upper_purple = np.array([170, 255, 255])
        purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
        processed_images.append(("Purple Filter", purple_mask))
        
        # Green range
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        processed_images.append(("Green Filter", green_mask))
    
    return processed_images

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

def fetch_captcha_code(fid, retry_queue=None):
    if retry_queue is None:
        retry_queue = {}
        
    attempts = 0
    current_time = time.time()
    first_attempt_success = False
    retry_needed = False
    
    while attempts < MAX_CAPTCHA_ATTEMPTS:
        counters["captcha_attempts"] += 1
        
        payload = encode_data({"fid": fid, "time": int(time.time() * 1000), "init": "0"})
        response = make_request(CAPTCHA_URL, payload)
        if response and response.status_code == 200:
            try:
                captcha_data = response.json()
                if captcha_data.get("code") == 1 and captcha_data.get("msg") == "CAPTCHA GET TOO FREQUENT.":
                    log("Captcha fetch too frequent, adding to retry queue...")
                    retry_queue[fid] = current_time + CAPTCHA_SLEEP
                    return None, retry_queue

                if "data" in captcha_data and "img" in captcha_data["data"]:
                    img_field = captcha_data["data"]["img"]
                    if img_field.startswith("data:image"):
                        img_base64 = img_field.split(",", 1)[1]
                    else:
                        img_base64 = captcha_data["data"]["img"]
                    img_bytes = base64.b64decode(img_base64)

                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    processed_images = preprocess_captcha(img_np)
                                        
                    # Try OCR on all processed images and evaluate results
                    candidates = []
                    for method_name, processed_img in processed_images:
                        if len(processed_img.shape) == 3:
                            if processed_img.shape[2] == 3:
                                processed_bytes = cv2.imencode('.png', processed_img)[1].tobytes()
                            else:
                                processed_bytes = cv2.imencode('.png', processed_img)[1].tobytes()
                        else:
                            processed_bytes = cv2.imencode('.png', processed_img)[1].tobytes()
                            
                        results = reader.readtext(processed_bytes, detail=1)
                        for result in results:
                            if len(result) >= 2:
                                text = result[1].strip().replace(' ', '')
                                confidence = result[2]
                                if text and confidence > MIN_CONFIDENCE:
                                    candidates.append((text, confidence, method_name))
                    
                    # Find candidate with the highest confidence and 4 alpanumerics
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    best_result = ""
                    for text, confidence, method_name in candidates:
                        if text.isalnum() and len(text) == 4:
                            best_result = text
                            log(f"Found valid captcha with method: {method_name}, confidence: {confidence:.2f}")
                            break
                    
                    if not best_result:
                        log("OCR did not return a valid result, requesting new captcha")
                        if attempts > 0:
                            retry_needed = True
                        attempts += 1
                        time.sleep(random.uniform(1.0, 3.0))
                        continue
                    
                    if attempts == 0:
                        first_attempt_success = True
                        
                    captcha_code = best_result

                    if getattr(args, 'all_images', False) or not (captcha_code.isalnum() and len(captcha_code) == 4):
                        save_captcha_image(img_np, fid, attempts, captcha_code)

                    if captcha_code.isalnum() and len(captcha_code) == 4:
                        log(f"Recognized captcha: {captcha_code}")
                        counters["captcha_success"] += 1
                        if first_attempt_success:
                            counters["captcha_first_try"] += 1
                        if attempts > 0:
                            counters["captcha_retry"] += attempts
                        return captcha_code, retry_queue
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

        if attempts > 0:
            retry_needed = True
        attempts += 1
        time.sleep(random.uniform(1.0, 3.0))

    log(f"Failed to fetch valid captcha after {attempts} attempts, adding to retry queue")
    counters["captcha_failures"] += 1
    retry_queue[fid] = current_time + CAPTCHA_SLEEP
    return None, retry_queue

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

def redeem_gift_code(fid, cdk, retry_queue=None):
    if retry_queue is None:
        retry_queue = {}
        
    if not str(fid).strip().isdigit():
        log(f"Skipping invalid FID: '{fid}'")
        return {"msg": "Invalid FID format"}, retry_queue
    fid = str(fid).strip()
    
    current_time = time.time()
    if fid in retry_queue and retry_queue[fid] > current_time:
        cooldown_remaining = int(retry_queue[fid] - current_time)
        log(f"FID {fid} is in cooldown period. {cooldown_remaining} seconds remaining.")
        return {"msg": "In cooldown"}, retry_queue

    try:
        login_payload = encode_data({"fid": fid, "time": int(time.time() * 1000)})
        login_resp = make_request(LOGIN_URL, login_payload)
        if not login_resp:
            return {"msg": "Login request failed after retries"}, retry_queue

        login_data = login_resp.json()
        if login_data.get("code") != 0:
            login_msg = login_data.get('msg', 'Unknown login error')
            log(f"Login failed for {fid}: {login_data.get('code')}, {login_msg}")
            return {"msg": f"Login failed: {login_msg}"}, retry_queue

        nickname = login_data.get("data", {}).get("nickname")
        fid_index = all_player_ids.index(fid) + 1 if fid in all_player_ids else 0
        log(f"Processing {nickname or 'Unknown Player'} ({fid}) [{fid_index}/{len(all_player_ids)}]")

        for attempt in range(CAPTCHA_RETRIES):
            try:
                captcha_code, retry_queue = fetch_captcha_code(fid, retry_queue)
                
                if captcha_code is None:
                    return {"msg": "Added to retry queue due to captcha rate limit"}, retry_queue
                
                redeem_payload = encode_data({
                    "fid": fid,
                    "cdk": cdk,
                    "captcha_code": captcha_code,
                    "time": int(time.time() * 1000)
                })
                redeem_resp = make_request(REDEEM_URL, redeem_payload)
                if not redeem_resp:
                    return {"msg": "Redemption request failed after retries"}, retry_queue

                redeem_data = redeem_resp.json()
                msg = redeem_data.get('msg', 'Unknown error').strip('.')

                if msg in ["CAPTCHA CHECK ERROR", "Sign Error", "Server requested retry"]:
                    log(f"Captcha attempt failed, retrying in a bit... (Attempt {attempt+1}/{CAPTCHA_RETRIES})")
                    time.sleep(random.uniform(2.5, 4))
                    continue
                elif msg == "CAPTCHA CHECK TOO FREQUENT":
                    log(f"Captcha check too frequent, adding to retry queue for {CAPTCHA_SLEEP} seconds...")
                    retry_queue[fid] = current_time + CAPTCHA_SLEEP
                    return {"msg": "Added to retry queue"}, retry_queue
                elif msg == "NOT LOGIN":
                    log(f"Session expired or invalid after captcha. Skipping {fid}.")
                    return {"msg": "Session expired after captcha"}, retry_queue
                else:
                    return redeem_data, retry_queue

            except Exception as e:
                log(f"Error during captcha attempt {attempt+1}: {str(e)}")
                if attempt == CAPTCHA_RETRIES - 1:
                    log(f"Reached max captcha retries for FID {fid}, adding to retry queue")
                    retry_queue[fid] = current_time + CAPTCHA_SLEEP
                    return {"msg": "Max retries reached, added to retry queue"}, retry_queue

        retry_queue[fid] = current_time + CAPTCHA_SLEEP
        return {"msg": "Added to retry queue"}, retry_queue

    except Exception as e:
        log(f"Unexpected error during redemption for {fid}: {str(e)}")
        return {"msg": f"Unexpected Error: {str(e)}"}, retry_queue

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
    script_end_time = time.time()
    total_seconds = script_end_time - script_start_time
    execution_time = str(timedelta(seconds=int(total_seconds)))
    
    log("\n=== Redemption Complete ===")
    log(f"Successfully redeemed: {counters['success']}")
    log(f"Already redeemed: {counters['already_redeemed']}")
    log(f"Errors/Failures: {counters['errors']}")
    
    if error_details:
        log("\n=== Error Details ===")
        log("The following IDs encountered errors:")
        for fid, error_msg in error_details.items():
            log(f"FID {fid}: {error_msg}")
    
    log("\n=== Captcha Statistics ===")
    log(f"Total attempts: {counters['captcha_attempts']}")
    log(f"Successful decodes: {counters['captcha_success']}")
    log(f"First attempt success: {counters['captcha_first_try']}")
    log(f"Retries: {counters['captcha_retry']}")
    log(f"Complete failures: {counters['captcha_failures']}")
    
    success_rate = 0
    if counters['captcha_attempts'] > 0:
        success_rate = (counters['captcha_success'] / counters['captcha_attempts']) * 100
        
    first_try_rate = 0
    if counters['captcha_success'] > 0:
        first_try_rate = (counters['captcha_first_try'] / counters['captcha_success']) * 100
        
    avg_attempts = 0
    if counters['captcha_success'] > 0:
        avg_attempts = (counters['captcha_attempts'] / counters['captcha_success'])
    
    log(f"Success rate: {success_rate:.2f}%")
    log(f"First try success rate: {first_try_rate:.2f}%")
    log(f"Average attempts per successful captcha: {avg_attempts:.2f}")
    log(f"Total execution time: {execution_time}")

if __name__ == "__main__":
    # Log initialization message
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"\n=== Starting redemption for gift code: {args.code} at {start_time} ===")

    # Handle *.csv input
    if args.csv == "*.csv":
        csv_files = glob(os.path.join(script_dir, "*.csv"))
    else:
        if os.path.isdir(args.csv):
            csv_files = glob(os.path.join(args.csv, "*.csv"))
        else:
            csv_files = [args.csv]

    if not csv_files:
        log("Error: No CSV files found.")
        sys.exit(1)

    # Process all CSV files
    retry_queue = {}
    all_player_ids = []
    
    # Load all player IDs first
    for csv_file in csv_files:
        try:
            player_ids = read_player_ids_from_csv(csv_file)
            log(f"Loaded {len(player_ids)} player IDs from {csv_file}")
            all_player_ids.extend(player_ids)
        except FileNotFoundError:
            log(f"Error: CSV file '{csv_file}' not found")
        except Exception as e:
            log(f"Error processing {csv_file}: {str(e)}")
    
    log(f"Total of {len(all_player_ids)} player IDs to process")
    
    # Continue processing until all FIDs are done
    processed_fids = set()
    stuck_counter = 0  # To detect when we're stuck
    max_stuck_loops = 10  # Maximum number of loops with no progress before we exit
    retry_attempts = {}  # Track number of retry attempts for each FID
    MAX_RETRY_ATTEMPTS = 5  # Maximum number of times we'll retry a FID before giving up

    while len(processed_fids) < len(all_player_ids):
        current_time = time.time()
        initial_processed_count = len(processed_fids)
        
        # Process regular queue
        for fid in all_player_ids:
            if fid in processed_fids:
                continue
                
            # Skip FIDs in cooldown
            if fid in retry_queue and retry_queue[fid] > current_time:
                continue
                
            # Initialize retry counter for this FID if not exists
            if fid not in retry_attempts:
                retry_attempts[fid] = 0
                
            result, retry_queue = redeem_gift_code(fid, args.code, retry_queue)
            
            raw_msg = result.get('msg', 'Unknown error').strip('.')
            friendly_msg = RESULT_MESSAGES.get(raw_msg, raw_msg)
            
            # Check if this FID was put in the retry queue
            if raw_msg in ["Added to retry queue", "Max retries reached, added to retry queue", 
                        "In cooldown", "Added to retry queue due to captcha rate limit"]:
                retry_attempts[fid] += 1
                
                if retry_attempts[fid] >= MAX_RETRY_ATTEMPTS:
                    log(f"FID {fid} failed after {MAX_RETRY_ATTEMPTS} retry attempts, marking as error")
                    processed_fids.add(fid)
                    counters["errors"] += 1
                    error_details[fid] = f"Failed after {MAX_RETRY_ATTEMPTS} retry attempts"
                    if fid in retry_queue:
                        del retry_queue[fid]
                else:
                    log(f"FID {fid} will be retried later: {friendly_msg} (Attempt {retry_attempts[fid]}/{MAX_RETRY_ATTEMPTS})")
                continue
                
            # Mark as processed if not in retry queue
            if raw_msg not in ["Added to retry queue", "Max retries reached, added to retry queue", 
                            "In cooldown", "Added to retry queue due to captcha rate limit"]:
                processed_fids.add(fid)
            
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
            elif raw_msg in ["TIMEOUT RETRY", "Server requested retry"]:
                pass
            else:
                counters["errors"] += 1
                error_details[fid] = friendly_msg
            
            log(f"Result: {friendly_msg}")
            time.sleep(DELAY)
        
        # Check if we have any FIDs waiting in the retry queue
        waiting_fids = [f for f, t in retry_queue.items() if t > current_time and f not in processed_fids]
        if waiting_fids:
            next_retry_time = min([t for f, t in retry_queue.items() if f in waiting_fids])
            wait_time = max(1, min(5, next_retry_time - current_time))
            
            log(f"{len(waiting_fids)} FIDs in cooldown. Next retry in ~{int(wait_time)} seconds.")
            log(f"Progress: {len(processed_fids)}/{len(all_player_ids)} FIDs processed")
            time.sleep(wait_time)
        elif len(processed_fids) < len(all_player_ids):
            # Double check if there are any FIDs that need processing but aren't in retry queue
            remaining = set(all_player_ids) - processed_fids
            remaining_not_in_queue = [fid for fid in remaining if fid not in retry_queue]
            
            if remaining_not_in_queue:
                log(f"Found {len(remaining_not_in_queue)} FIDs that need processing but aren't in retry queue")
                continue
            else:
                # We might be stuck - check if we're making progress
                if len(processed_fids) == initial_processed_count:
                    stuck_counter += 1
                    log(f"No progress made in this loop. Stuck counter: {stuck_counter}/{max_stuck_loops}")
                    
                    if stuck_counter >= max_stuck_loops:
                        log("No progress after multiple attempts. Some FIDs may be stuck in retry queue.")
                        log("Checking remaining FIDs for final processing attempt...")
                        
                        # Attempt to process each remaining FID one last time
                        for fid in remaining:
                            log(f"Final processing attempt for FID: {fid}")
                            # Reset retry queue entry to allow immediate processing
                            if fid in retry_queue:
                                del retry_queue[fid]
                            
                            # Force one last attempt
                            result, _ = redeem_gift_code(fid, args.code, {})
                            raw_msg = result.get('msg', 'Unknown error').strip('.')
                            friendly_msg = RESULT_MESSAGES.get(raw_msg, raw_msg)
                            
                            # Mark as processed regardless of result
                            processed_fids.add(fid)
                            
                            # Update counters based on final result
                            if raw_msg in ["SUCCESS", "SAME TYPE EXCHANGE"]:
                                counters["success"] += 1
                            elif raw_msg == "RECEIVED":
                                counters["already_redeemed"] += 1
                            elif raw_msg in ["TIMEOUT RETRY", "Server requested retry"]:
                                counters["errors"] += 1
                                error_details[fid] = "Final attempt: Server timeout/retry"
                            else:
                                counters["errors"] += 1
                                error_details[fid] = f"Final attempt: {friendly_msg}"
                            
                            log(f"Final result for {fid}: {friendly_msg}")
                            time.sleep(DELAY)
                        
                        break
                else:
                    # We made progress, reset the counter
                    stuck_counter = 0
                    
                # Put a small delay to avoid tight looping
                time.sleep(5)
                continue
    
    # Ensure all FIDs in retry queue are processed before exiting
    remaining_in_queue = [fid for fid in all_player_ids if fid in retry_queue and fid not in processed_fids]
    if remaining_in_queue:
        log(f"Still have {len(remaining_in_queue)} FIDs in retry queue. Processing them before exiting...")
        
        # Process remaining FIDs in retry queue until they're all done
        while remaining_in_queue:
            current_time = time.time()
            ready_fids = [fid for fid in remaining_in_queue if retry_queue.get(fid, 0) <= current_time]
            
            if ready_fids:
                for fid in ready_fids:
                    log(f"Processing remaining queued FID: {fid}")
                    result, retry_queue = redeem_gift_code(fid, args.code, retry_queue)
                    
                    raw_msg = result.get('msg', 'Unknown error').strip('.')
                    friendly_msg = RESULT_MESSAGES.get(raw_msg, raw_msg)
                    
                    # Check if we should mark it as processed
                    if raw_msg not in ["Added to retry queue", "Max retries reached, added to retry queue", 
                                     "In cooldown", "Added to retry queue due to captcha rate limit"]:
                        processed_fids.add(fid)
                        remaining_in_queue.remove(fid)
                    
                    # Update counters based on result
                    if raw_msg in ["SUCCESS", "SAME TYPE EXCHANGE"]:
                        counters["success"] += 1
                    elif raw_msg == "RECEIVED":
                        counters["already_redeemed"] += 1
                    elif raw_msg in ["TIMEOUT RETRY", "Server requested retry"]:
                        pass
                    else:
                        counters["errors"] += 1
                        error_details[fid] = friendly_msg
                    
                    log(f"Result: {friendly_msg}")
                    time.sleep(DELAY)
            else:
                # Wait for the next FID to be ready
                next_ready_time = min([retry_queue.get(fid, float('inf')) for fid in remaining_in_queue])
                wait_time = max(1, min(10, next_ready_time - current_time))
                log(f"Waiting {int(wait_time)}s for next FID in queue to be ready...")
                time.sleep(wait_time)
            
            # Refresh the list after processing
            remaining_in_queue = [fid for fid in all_player_ids if fid in retry_queue and fid not in processed_fids]
    
    # Make one final check for unprocessed FIDs
    unprocessed_fids = set(all_player_ids) - processed_fids
    if unprocessed_fids:
        log(f"Found {len(unprocessed_fids)} FIDs that still need to be processed. Handling them now...")
        
        # Process any remaining FIDs that somehow got missed
        for fid in unprocessed_fids:
            log(f"Final processing attempt for FID: {fid}")
            result, _ = redeem_gift_code(fid, args.code, {})
            
            raw_msg = result.get('msg', 'Unknown error').strip('.')
            friendly_msg = RESULT_MESSAGES.get(raw_msg, raw_msg)
            
            # Update counters based on result
            if raw_msg in ["SUCCESS", "SAME TYPE EXCHANGE"]:
                counters["success"] += 1
            elif raw_msg == "RECEIVED":
                counters["already_redeemed"] += 1
            elif raw_msg in ["TIMEOUT RETRY", "Server requested retry"]:
                pass
            else:
                counters["errors"] += 1
                error_details[fid] = friendly_msg
            
            log(f"Final result for {fid}: {friendly_msg}")
            processed_fids.add(fid)
            time.sleep(DELAY)
    
    log(f"All {len(processed_fids)}/{len(all_player_ids)} FIDs processed.")
    print_summary()