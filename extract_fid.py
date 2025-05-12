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

default_ocr = "paddleocr" if PADDLEOCR_AVAILABLE else ocr_choices[0]

parser.add_argument("--input", required=True, help="Path to the input video file.")
parser.add_argument("--ocr-method", choices=ocr_choices, default=default_ocr,
                    help=f"OCR method to use. Available: {', '.join(ocr_choices)}. (Default: {default_ocr})")
parser.add_argument("--output-file", default="output.csv", help="Path to the output CSV file (default: output.csv).")
parser.add_argument("--enable-denoising", action="store_true", help="Enable denoising of the grayscale image.")
parser.add_argument("--frame-skip", type=int, default=10, help="Process every nth frame (default: 10).")
args = parser.parse_args()

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

def process_with_easyocr(frame):
    """Process the frame using EasyOCR and return extracted text."""
    if not hasattr(process_with_easyocr, "reader"):
        process_with_easyocr.reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader once
    # Convert to grayscale for EasyOCR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if args.enable_denoising:
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    result = process_with_easyocr.reader.readtext(gray, detail=0, allowlist='0123456789')
    return ''.join(result).strip()  # Return the extracted text


def process_with_paddleocr(frame):
    """Process the frame using PaddleOCR and return a 9-digit ID if found."""
    if not hasattr(process_with_paddleocr, "paddle_ocr"):
        process_with_paddleocr.paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)  # Initialize PaddleOCR once
    # Apply denoising for PaddleOCR (BGR format) if enabled
    if args.enable_denoising:
        frame = cv2.fastNlMeansDenoisingColored(frame, None, h=10, templateWindowSize=7, searchWindowSize=21)
    result = process_with_paddleocr.paddle_ocr.ocr(frame, cls=False)
    if result and isinstance(result[0], list):
        for line in result[0]:
            text = line[1][0]
            id_match = re.search(r'\b\d{9}\b', text)  # Extract 9-digit IDs
            if id_match:
                return id_match.group(0)  # Return the first matched ID
    return None


def process_with_pytesseract(frame):
    """Process the frame using PyTesseract and return extracted text."""
    # Convert to grayscale for PyTesseract
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply denoising for PyTesseract if enabled
    if args.enable_denoising:
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    text = pytesseract.image_to_string(gray, config='--psm 13 -c tessedit_char_whitelist=0123456789')
    return text.strip()  # Return the extracted text


# Main processing logic
extracted_numbers = set()
count = 0

# Process the video frame by frame
print(f"Processing video '{args.input}' using {args.ocr_method}...")

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

    # Perform OCR using the selected library
    text = ""
    if args.ocr_method == "easyocr" and EASYOCR_AVAILABLE:
        text = process_with_easyocr(roi)
    elif args.ocr_method == "paddleocr" and PADDLEOCR_AVAILABLE:
        text = process_with_paddleocr(roi)
    elif args.ocr_method == "pytesseract" and PYTESSERACT_AVAILABLE:
        text = process_with_pytesseract(roi)

    # Add the extracted text or ID to the set
    if text:
        extracted_numbers.add(text)

    # Print progress: frames processed and IDs found so far
    print(f"\rFrames Processed: {count + 1}, IDs Found: {len(extracted_numbers)}", end="", flush=True)

    count += 1

# End of video processing
print(f"\nFinished processing {count} frames.")

# Filter out non-numeric values and keep only 9-digit numbers
extracted_numbers = {num for num in extracted_numbers if len(num) == 9 and num.isdigit()}

# Write the extracted numbers to a CSV file
with open(args.output_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for number in sorted(extracted_numbers):
        writer.writerow([number])

print(f"\nExtracted numbers saved to {args.output_file}")
print(f"Number of unique numbers: {len(extracted_numbers)}")

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()