# Script to extract whiteout survival player ids from a video using OCR 
# which can then be used by redeem_codes.py to obtain gifts!
#
# Sample usage:
# python extract_fid.py --input video.mp4 --ocr-method easyocr --output-file output.csv --enable-denoising
# This script is designed to extract player IDs from a video of the 
# game "Whiteout Survival" using Optical Character Recognition (OCR).
#
# It supports multiple OCR libraries: PaddleOCR, EasyOCR, and Tesseract. PaddleOCR 
# is the default if available. In terms of quality, the order is PaddleOCR > Tesseract > EasyOCR
#
# Practical usage:
#
# I play on iOS, so I use iCloud to just download the video to my computer, this gives 
# me the 592x1280px resolution. I use the screen recording tool on iOS to record.
# If you are using a different resolution, grab a frame still from the video and
# adjust the x, y, w, h values in the script to match the player ID location.
#
# 1. Make a video of the game "Whiteout Survival" showing the player IDs
#    Use a screen recording tool to capture the video, default resolution I 
#    used is 592x1280px.

#    This is easiest by being in the alliance, going to the alliance tab, and scrolling 
#    through the members clicking on their names spend about half a second on each 
#    member to ensure the OCR can read it
#   
# 2. Upload the file to iCloud and generate a link then download it and run through script
# 3. Take the resulting CSV and feed it to the redeem_codes.py script
# 4. Enjoy the rewards!

import os
import cv2
import argparse
import csv
import re
import logging
from typing import Optional, Set

# Configure logging
# logging.basicConfig(filename="extract_fid.log", level=logging.INFO, format="%(levelname)s: %(message)s")

# Dynamically check for OCR library availability
PADDLEOCR_AVAILABLE = False
EASYOCR_AVAILABLE = False
PYTESSERACT_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    print("Warning: PaddleOCR library not found.")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    print("Warning: EasyOCR library not found.")

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update this path as needed
except ImportError:
    print("Warning: Tesseract library not found.")

# Set up logging configuration - Paddle OCR hijacks the root logger and we need to reset it to take back control within the application
# Remove all existing handlers from the root logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up the logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("extract_fid.log"),
        logging.StreamHandler()
    ]
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Extract numbers from video using OCR.")
ocr_choices = []
if PADDLEOCR_AVAILABLE:
    ocr_choices.append("paddleocr")
if EASYOCR_AVAILABLE:
    ocr_choices.append("easyocr")
if PYTESSERACT_AVAILABLE:
    ocr_choices.append("pytesseract")

if not ocr_choices:
    print("CRITICAL ERROR: No OCR libraries found. Please install PaddleOCR, EasyOCR, or Tesseract.")
    exit(1)

ocr_choices.append("all")  # Add the 'all' option to run all OCR mechanisms

default_ocr = "paddleocr" if PADDLEOCR_AVAILABLE else ocr_choices[0]

parser.add_argument("--input", required=True, help="Path to the input video file.")
parser.add_argument("--ocr-method", choices=ocr_choices, default=default_ocr,
                    help=f"OCR method to use. Available: {', '.join(ocr_choices)}. (Default: {default_ocr})")
parser.add_argument("--output-file", help="Path to the output CSV file. If not specified, the input filename will be used with a .csv extension.")
parser.add_argument("--enable-denoising", action="store_true", help="Enable denoising of the grayscale image.")
parser.add_argument("--frame-skip", type=int, default=10, help="Process every nth frame (default: 10).")
args = parser.parse_args()

# Determine the output file name
if not args.output_file:
    # Replace .mp4 with .csv in the input filename
    if args.input.lower().endswith(".mp4"):
        args.output_file = args.input[:-4] + ".csv"
    else:
        args.output_file = args.input + ".csv"

print(f"Output file will be saved as: {args.output_file}")

# Check if the input file exists
if not os.path.exists(args.input):
    print(f"ERROR: Input file '{args.input}' does not exist.")
    exit(1)

# Define the video capture object
cap = cv2.VideoCapture(args.input)

# Get the frame width and height of the input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the default resolution and scaling factors
default_width = 592
default_height = 1280
width_ratio = frame_width / default_width
height_ratio = frame_height / default_height

# Define the region of interest (ROI) coordinates and scale them based on the video resolution
x, y, w, h = 240, 971, 126, 26  # Default ROI for 592x1280 resolution
x = round(x * width_ratio)
y = round(y * height_ratio)
w = round(w * width_ratio)
h = round(h * height_ratio)

print(f"Adjusted ROI coordinates to x={x}, y={y}, w={w}, h={h} based on video resolution {frame_width}x{frame_height}.")


def filter_9_digit_numbers(text):
    """
    Helper function to filter and return 9-digit numbers from text.
    
    This function takes a string or list of strings and filters out any text
    that is not exactly 9 digits long.
    """
    if isinstance(text, list):
        text = '\n'.join(text)  # Convert list to a single string
    return [line.strip() for line in text.splitlines() if len(line.strip()) == 9 and line.strip().isdigit()]


def process_with_easyocr(frame):
    """
    Process the frame using EasyOCR and return extracted text.
    
    This method uses EasyOCR to extract text from the given frame. It filters the results
    to include only 9-digit numbers. If denoising is enabled, it applies denoising to the
    grayscale image before OCR processing.
    """
    if not hasattr(process_with_easyocr, "reader"):
        process_with_easyocr.reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader once

    # Convert to grayscale for EasyOCR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if args.enable_denoising:
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    result = process_with_easyocr.reader.readtext(gray, detail=0, allowlist='0123456789')

    # Filter results to include only 9-digit numbers
    filtered_result = filter_9_digit_numbers(result)
    return ''.join(filtered_result) if filtered_result else None  # Return the first valid result or None


def process_with_paddleocr(frame):
    """
    Process the frame using PaddleOCR and return a 9-digit ID if found.
    
    This method uses PaddleOCR to extract text from the given frame. It applies denoising
    if enabled and filters the results to include only 9-digit numbers using regex.
    """
    if not hasattr(process_with_paddleocr, "paddle_ocr"):
        process_with_paddleocr.paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)  # Initialize PaddleOCR once

    # Apply denoising for PaddleOCR (BGR format) if enabled
    if args.enable_denoising:
        frame = cv2.fastNlMeansDenoisingColored(frame, None, h=10, templateWindowSize=7, searchWindowSize=21)

    result = process_with_paddleocr.paddle_ocr.ocr(frame, cls=False)

    if result and isinstance(result[0], list):
        text = [line[1][0] for line in result[0]]  # Extract text from PaddleOCR results
        filtered_result = filter_9_digit_numbers(text)
        return ''.join(filtered_result) if filtered_result else None  # Return the first valid result or None

    return None


def process_with_pytesseract(frame):
    """
    Process the frame using PyTesseract and return extracted text.
    
    This method uses PyTesseract to extract text from the given frame. It filters the results
    to include only 9-digit numbers. If denoising is enabled, it applies denoising to the
    grayscale image before OCR processing.
    """
    # Convert to grayscale for PyTesseract
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply denoising for PyTesseract if enabled
    if args.enable_denoising:
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    result = pytesseract.image_to_string(gray, config='--psm 13 -c tessedit_char_whitelist=0123456789')

    # Filter results to include only 9-digit numbers
    filtered_result = filter_9_digit_numbers(result)
    return ''.join(filtered_result) if filtered_result else None  # Return the first valid result or None


def process_with_ocr(roi, method):
    """
    Process the frame using the specified OCR method and return extracted text.
    
    This method dynamically selects the OCR engine (EasyOCR, PaddleOCR, or PyTesseract)
    based on the provided method and processes the given region of interest (ROI).
    """
    if method == "easyocr" and EASYOCR_AVAILABLE:
        return process_with_easyocr(roi)
    elif method == "paddleocr" and PADDLEOCR_AVAILABLE:
        return process_with_paddleocr(roi)
    elif method == "pytesseract" and PYTESSERACT_AVAILABLE:
        return process_with_pytesseract(roi)
    return None


def check_ocr_availability(method: str) -> bool:
    """
    Check if the specified OCR method is available.
    """
    if method == "easyocr":
        return EASYOCR_AVAILABLE
    elif method == "paddleocr":
        return PADDLEOCR_AVAILABLE
    elif method == "pytesseract":
        return PYTESSERACT_AVAILABLE
    return False


def report_progress(count: int, total_ids: int) -> None:
    """
    Report the progress of frame processing.
    """
    print(f"Frames Processed: {count + 1}, IDs Found: {total_ids}", end="\r", flush=True)


def process_video_with_method(cap: cv2.VideoCapture, method: str, extracted_numbers: Set[str]) -> None:
    """
    Process the video frame by frame using the specified OCR method.

    Args:
        cap: The video capture object.
        method: The OCR method to use.
        extracted_numbers: A set to store extracted numbers.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the first frame
    count = 0

    logging.info(f"Processing video '{args.input}' using {method}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames based on the frame-skip parameter
        if count % args.frame_skip != 0:
            count += 1
            continue

        # Extract the region of interest
        roi = frame[y:y + h, x:x + w]

        # Perform OCR using the specified method
        text = process_with_ocr(roi, method)
        if text:
            extracted_numbers.add(text)

        # Report progress
        report_progress(count, len(extracted_numbers))

        count += 1

    logging.info(f"Finished processing {count} frames with {method}. Detected {len(extracted_numbers)} IDs.")

# Define the extracted_numbers set to store unique IDs
extracted_numbers = set()

# Run the OCR process
if args.ocr_method == "all":
    for method in ["easyocr", "paddleocr", "pytesseract"]:
        if check_ocr_availability(method):
            process_video_with_method(cap, method, extracted_numbers)
        else:
            logging.warning(f"OCR method '{method}' is not available. Skipping...")
else:
    if not check_ocr_availability(args.ocr_method):
        logging.error(f"OCR method '{args.ocr_method}' is not available. Please install the required library.")
        exit(1)

    # If the selected OCR method is available, process the video
    process_video_with_method(cap, args.ocr_method, extracted_numbers)

# Filter out non-numeric values and keep only 9-digit numbers
extracted_numbers = {num for num in extracted_numbers if len(num) == 9 and num.isdigit()}

# Handle empty results
if not extracted_numbers:
    logging.warning("No valid IDs were extracted from the video.")
else:
    # Write the extracted numbers to a CSV file
    with open(args.output_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for number in sorted(extracted_numbers):
            writer.writerow([number])

    logging.info(f"Extracted numbers saved to {args.output_file}")
    logging.info(f"Number of unique numbers: {len(extracted_numbers)}")

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()