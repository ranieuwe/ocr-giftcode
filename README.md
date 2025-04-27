# Whiteout Survival Gift Code Redemption Script

Python script that automates the process of redeeming gift codes in the game **Whiteout Survival**. It reads a list of Player IDs from one or more `.csv` files, automatically attempts to solve the CAPTCHA required by the API using OCR, sends requests to the game's giftcode redemption API, and redeems the specified gift code for each player.

---

## Features

- **Flexible CSV Import**: Reads player IDs from `.csv` files. Supports **both** formats:
    - One player ID per line.
    - Multiple player IDs on a single line, separated by commas (whitespace is ignored).
- **Automatic CAPTCHA Solving**: Uses OCR (EasyOCR) to automatically read and submit CAPTCHA codes required by the redemption API.
- **Command-Line Interface**: Accepts the `.csv` file path (or a directory, or `*.csv`) and gift code as arguments.
- **Error Handling**: Provides clear error messages for missing/invalid inputs, API responses, and CAPTCHA failures.
- **Retry Logic**: Automatically retries failed requests when the server indicates temporary issues (e.g., `TIMEOUT RETRY`) or CAPTCHA errors.
- **Verbose Logging**: Shows timestamp, player nickname (if available), ID, and CAPTCHA status during processing. Logs all output to a file (`redeemed_codes.txt`).
- **CAPTCHA Image Logging**: Saves attempted CAPTCHA images to a unique folder (`failed_captchas_YYYYMMDD_HHMMSS`) for debugging purposes.
- **GPU Acceleration**: Optionally enable GPU for faster OCR processing if available (`--use-gpu` flag).
- **Save All Captcha Images**: Optionally save every fetched CAPTCHA image regardless of OCR success (`--all-images` flag).
- **Summary Report**: Provides a summary of successful redemptions, already redeemed codes, and errors at the end.
- **Rate Limiting**: Includes a configurable delay between requests to avoid triggering API rate limits, with specific handling for CAPTCHA frequency limits.

---

## Prerequisites

1.  **Python 3.x**: Download and install Python from [python.org](https://www.python.org/). Ensure Python is added to your system's PATH during installation.
2.  **Required Libraries**: Install the necessary Python libraries using `pip` from your command line or terminal:
    ~~~bash
    pip install requests easyocr opencv-python numpy torch torchvision torchaudio
    ~~~

---

## Setup

1.  **Get the Script**:
    *   Clone the repository:
        ~~~bash
        git clone https://github.com/justncodes/wos-giftcode.git
        cd wos-giftcode
        ~~~
    *   Or, download the `redeem_codes.py` script directly from the repository.

2.  **Install Dependencies**: Run the `pip install ...` command listed in the **Prerequisites** section.

3.  **Prepare the CSV File(s)**:
    *   Create one or more `.csv` files (e.g., `player_ids.csv`).
    *   **Format**: You can list IDs in two ways (or even mix formats within a file, though keeping it consistent is recommended):
        *   **One ID per line:** Each player ID is on its own row.
        *   **Comma-separated IDs:** Multiple player IDs on the same line, separated by commas. Whitespace around the commas or IDs will be automatically removed.
    *   Place the `.csv` file(s) in the same directory as the script, or note the full path to provide as an argument.

    *   **Example `ids_newline.csv` (One ID per line):**
        ~~~csv
        57845354
        98765432
        12345678
        ~~~

    *   **Example `ids_comma.csv` (Comma-separated):**
        ~~~csv
        57845354, 98765432, 12345678
        44445555, 66667777
        ~~~

    *   **Example `ids_mixed.csv` (Mixed - also works):**
        ~~~csv
        57845354
        98765432, 12345678
        44445555,66667777,88889999
        11112222
        ~~~

---

## Usage

Run the script from your command line or terminal using `python`:

~~~bash
python redeem_codes.py --csv <path_or_pattern> --code <gift_code>
~~~

### Arguments

-   `--csv`: Specifies the player ID source. Can be:
    *   A path to a single `.csv` file (e.g., `player_ids.csv` or `C:\\Users\\You\\Documents\\ids.csv`).
    *   A path to a folder/directory containing `.csv` files (e.g., `/home/user/wos_ids/` or `.` for the current directory). The script will process all files ending in `.csv` within that folder.
    *   The pattern `*.csv` to process all `.csv` files located in the *same directory as the script itself*.
-   `--code`: The gift code you want to redeem (e.g., `WOS2025`).
-   `--all-images`: Save all CAPTCHA images regardless of OCR success.
-   `--use-gpu`: Enable GPU for EasyOCR reader.

### Examples

*   **Process a specific file in the current directory:**
    ~~~bash
    python redeem_codes.py --csv player_ids.csv --code ILoveWOS
    ~~~

*   **Process a specific file using its full path:**
    ~~~bash
    python redeem_codes.py --csv \"/path/to/your/data/player_ids.csv\" --code ILoveWOS
    ~~~

*   **Process all `.csv` files in a specific folder:**
    ~~~bash
    python redeem_codes.py --csv /path/to/your/folder --code ILoveWOS
    ~~~

*   **Process all `.csv` files located in the script's directory:**
    ~~~bash
    python redeem_codes.py --csv "*.csv" --code ILoveWOS
    ~~~

---

## Output

### Sample Run (including CAPTCHA messages)

~~~plaintext
Created directory for captcha images: failed_captchas
Using CPU. Note: This module is much faster with a GPU.
2025-04-27 17:12:38 - 
=== Starting redemption for gift code: mK6DNry4w at 2025-04-27 17:12:38 ===
2025-04-27 17:12:38 - Reading TSK.csv (detected format: newline)
2025-04-27 17:12:38 - Loaded 72 player IDs from TSK.csv
2025-04-27 17:12:39 - Processing ༒ᶦᵃᵐKԋαʅҽҽʂι༒ (46068074)
Saved captcha image: failed_captchas\fid46068074_try0_OCR_SMF_1745763160.png
2025-04-27 17:12:40 - Invalid captcha format: 'SMF', refetching...
Saved captcha image: failed_captchas\fid46068074_try1_OCR_~FXPA_1745763163.png
2025-04-27 17:12:43 - Invalid captcha format: '~FXPA', refetching...
Saved captcha image: failed_captchas\fid46068074_try2_OCR_81A_1745763167.png
2025-04-27 17:12:47 - Invalid captcha format: '81A', refetching...
2025-04-27 17:12:52 - Recognized captcha: fY9U
2025-04-27 17:12:53 - Result: Already redeemed
2025-04-27 17:12:54 - Processing Mr Litosqui (44500803)
2025-04-27 17:12:55 - Recognized captcha: nsSz
2025-04-27 17:12:55 - Result: Already redeemed
2025-04-27 17:12:57 - Processing ＴＨＥ ＷＯＬＶＥＲＩＮＥ (62737144)
2025-04-27 17:12:58 - Recognized captcha: dibN
2025-04-27 17:12:58 - Result: Successfully redeemed
2025-04-27 17:13:00 - Processing Samwell (145616970)
2025-04-27 17:13:00 - Recognized captcha: VQ4G
2025-04-27 17:13:01 - CAPTCHA CHECK ERROR, retrying... (Attempt 1/20)
2025-04-27 17:13:06 - Recognized captcha: 4ZtX
2025-04-27 17:13:07 - CAPTCHA CHECK ERROR, retrying... (Attempt 2/20)
2025-04-27 17:13:10 - Recognized captcha: Ppzs
2025-04-27 17:13:11 - Result: Successfully redeemed
... [Rest of the processing] ...
2025-02-21 10:30:20 - === Redemption Complete ===
... [Summary Report] ...
~~~

### Logging

All console output is automatically appended to a log file named `redeemed_codes.txt`, created in the same directory where the script is run.

Additionally, a new folder named `failed_captchas_YYYYMMDD_HHMMSS` (where the timestamp corresponds to the script run) will be created in the script's directory. This folder will contain images of the CAPTCHAs the script attempted to solve, along with the OCR result in the filename (e.g., `fid12345678_try0_OCR_ABCD_timestamp.png`). This is useful for debugging CAPTCHA recognition issues.

---

## Troubleshooting

1.  **`pip` command not found**: Ensure Python is installed correctly and its `Scripts` directory (Windows) or equivalent (macOS/Linux) is in your system's PATH environment variable.
2.  **ModuleNotFoundError**: Make sure you've installed *all* the required libraries (`pip install requests easyocr opencv-python numpy torch torchvision torchaudio`). Pay attention to any errors during installation.
3.  **Dependency Installation Issues**: `easyocr`, PyTorch, or `opencv-python` can sometimes be tricky to install. Consult their official documentation or search for OS-specific installation guides (e.g., installing C++ build tools might be needed). Ensure you have a compatible Python version.
4.  **CSV File Not Found**: Verify the path provided with `--csv` is correct. Check for typos. If using `*.csv` or a folder path, ensure `.csv` files actually exist there.
5.  **Permission Denied (Log File / CAPTCHA Folder)**: Ensure the script has permission to write files and create directories in the location it's running from. Try running the terminal/command prompt as an administrator (use with caution).
6.  **Code Expired / Claim Limit Reached**: The script will detect these specific API responses (`TIME ERROR`, `USED`) and stop execution, logging the reason. Verify the gift code's validity.
7.  **API Rate Limiting / `TIMEOUT RETRY` failures**: If you see many `Server requested retry` messages followed by failures, the API might be temporarily overloaded or you might be sending requests too quickly. Try increasing the `DELAY` value (in seconds) near the top of the script file (e.g., `DELAY = 2`).
8.  **CAPTCHA Solving Failures (`CAPTCHA CHECK ERROR`, `Sign Error`)**: If the script frequently fails with CAPTCHA errors, even after retries:
    *   Check the saved images in the `failed_captchas_...` folder. Look at the filenames - does the `_OCR_XYZ_` part match the image? Are the images clear? Is the OCR consistently misreading them?
    *   The API's CAPTCHA might have changed its style (font, distortion, length), making it harder for EasyOCR to read.
    *   EasyOCR performance can vary. Ensure it's installed correctly. Using a GPU (if available and configured correctly with PyTorch) can improve speed and sometimes accuracy.
    *   If `CAPTCHA CHECK TOO FREQUENT` occurs often, the script will automatically sleep for `CAPTCHA_SLEEP` seconds, but persistent issues might indicate aggressive rate-limiting by the API.
9.  **EasyOCR Model Downloads**: The first time you run the script (or `easyocr`), it may need to download language models. Ensure you have an internet connection.

---

## Changelog

### v2.3.0 (Current Version)
- Integrated OCR (EasyOCR) to automatically read and submit CAPTCHA codes required by the redemption API.
- Added `easyocr`, `opencv-python`, `numpy`, and PyTorch (`torch`, `torchvision`, `torchaudio`) for OCR.
- Saves attempted CAPTCHA images with OCR results in the filename to a subfolder (`failed_captchas`) for debugging.
- Enhanced retry logic specifically for CAPTCHA-related errors (`CAPTCHA CHECK ERROR`, `Sign Error`, `CAPTCHA CHECK TOO FREQUENT`).

### v2.2.0
- Added support for reading comma-separated player IDs within CSV files.
- Improved CSV file path handling in `--csv` argument (folder path, `*.csv` pattern).
- Enhanced logging detail and error categorization.

### v2.1.0
- Added support for processing all .csv files in a folder or the script's directory using `*.csv`.
- Improved error handling for missing or invalid .csv files.

### v2.0.0
- Added retry logic for failed requests (e.g., `TIMEOUT RETRY`).
- Display player nicknames during processing.
- Added Unicode support for special characters and foreign alphabets.
- Added a summary report of redemptions at the end.
- Improved error handling and logging.

### v1.0.0 (Initial Release)
- Basic functionality: CSV import (newline separated), command-line args, API requests.
- Initial error handling and rate limiting.

---

## Credits

- **Author**: justncodes (\\[SIR\\] Yolo on #340)
- **Repository**: [wos-giftcode](https://github.com/justncodes/wos-giftcode)

---

## Support

If you encounter any issues or have questions, feel free to open an issue on the [GitHub repository](https://github.com/justncodes/wos-giftcode/issues). Please include relevant logs and, if the issue is CAPTCHA-related, mention if the saved images look correct.

---

## License

This project is licensed under the GPLv3 License. See the `LICENSE` file for details.
