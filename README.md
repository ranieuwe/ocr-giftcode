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
- **CAPTCHA Image Logging**: Optionally saves CAPTCHA images to a folder (`failed_captchas`) for debugging purposes.
- **GPU Acceleration**: Optional GPU support for faster OCR processing, with specific device selection (integrated or discrete GPU) via the `--use-gpu` flag.
- **Save All Captcha Images**: Optionally save every fetched CAPTCHA image regardless of OCR success (`--all-images` flag).
- **Summary Report**: Provides a summary of successful redemptions, already redeemed codes, errors and helpful statistics at the end.
- **Rate Limiting**: Includes a configurable delay between requests to avoid triggering API rate limits, with specific handling for CAPTCHA frequency limits.
- **Progress Tracking**: Shows current progress as each player ID is processed with current/total indicators.

---

## Prerequisites

1.  **Python 3.x**: Download and install Python from [python.org](https://www.python.org/). Ensure Python is added to your system's PATH during installation.
2.  **Required Libraries**: Install the necessary Python libraries using `pip` from your command line or terminal:
    ~~~bash
    pip install requests easyocr opencv-python numpy torch torchvision torchaudio pillow
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
python redeem_codes.py --csv <path_or_pattern> --code <gift_code> [options]
~~~

### Arguments

-   `--csv`: Specifies the player ID source. Can be:
    *   A path to a single `.csv` file (e.g., `player_ids.csv` or `C:\\Users\\You\\Documents\\ids.csv`).
    *   A path to a folder/directory containing `.csv` files (e.g., `/home/user/wos_ids/` or `.` for the current directory). The script will process all files ending in `.csv` within that folder.
    *   The pattern `*.csv` to process all `.csv` files located in the *same directory as the script itself*.
-   `--code`: The gift code you want to redeem (e.g., `WOS2025`).
-   `--all-images`: Save all CAPTCHA images regardless of OCR success.
-   `--use-gpu`: Enable GPU for EasyOCR reader. Optionally specify a device ID (e.g., `--use-gpu 0` for integrated GPU, `--use-gpu 1` for discrete GPU).

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

*   **Process with GPU acceleration (auto-select first available GPU):**
    ~~~bash
    python redeem_codes.py --csv "*.csv" --code ILoveWOS --use-gpu
    ~~~

*   **Process with specific GPU device (e.g., discrete GPU):**
    ~~~bash
    python redeem_codes.py --csv "*.csv" --code ILoveWOS --use-gpu 1
    ~~~

---

## GPU Support

Version 2.4.5 adds enhanced GPU support with the ability to select specific GPU devices, including discrete GPUs. This can significantly accelerate CAPTCHA recognition and improve overall script performance.

### How GPU Support Works

- The script uses PyTorch's GPU capabilities to run EasyOCR on your GPU.
- You can specify which GPU to use when you have multiple GPUs (integrated and discrete).
- Device IDs typically start at 0 (often the integrated GPU) with additional GPUs (like discrete/dedicated cards) at 1, 2, etc.

### Setting Up NVIDIA GPU Support

If you don't have a CUDA-capable GPU, the script will automatically fall back to CPU mode.

If you *do* have a CUDA-capable NVIDIA GPU and want to use it:

1. **Install the CUDA toolkit** (e.g. CUDA 11.8).
   * The CUDA toolkit provides the necessary libraries for GPU computing.
   * cuDNN (CUDA Deep Neural Network library) is an additional library that is often mentioned alongside CUDA, but is typically not required for basic PyTorch GPU usage with this script.
2. **Reinstall PyTorch with CUDA support**:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
   (Use the appropriate CUDA version number; cu118 = CUDA 11.8)

3. **Verify your GPU setup with Python**:
   ```python
   import torch
   print(torch.cuda.is_available(), torch.version.cuda) 
   print(torch.cuda.device_count(), "devices available")
   for i in range(torch.cuda.device_count()):
       print(f"Device {i}: {torch.cuda.get_device_name(i)}")
   ```
   You should see `True` and a CUDA version string, followed by your available GPU devices.

### Setting Up AMD GPU Support

EasyOCR can work with AMD GPUs using ROCm (Radeon Open Compute), though support is more limited compared to NVIDIA GPUs:

1. **Install ROCm** platform (follow instructions at the [ROCm Installation Guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)).
2. **Install PyTorch with ROCm support**:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2
   ```
   (Use the appropriate ROCm version; check compatibility with your GPU model)

3. **Verify your setup**:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should return True if ROCm is working
   print(torch.cuda.device_count(), "devices available")
   ```

5. **Important note**: AMD GPU support is more experimental and may not work with all AMD cards. The script should fall back to CPU mode if your AMD GPU isn't properly detected.

### Using the Script with GPU Support

- For auto-selection: `--use-gpu`
- For specific device: `--use-gpu 0` (often integrated) or `--use-gpu 1` (often discrete)

---

## Output

### Sample Run (including CAPTCHA messages)

~~~plaintext
Using GPU device 1: NVIDIA GeForce RTX 3080
2025-04-27 17:12:38 - 
=== Starting redemption for gift code: mK6DNry4w at 2025-04-27 17:12:38 ===
2025-04-27 17:12:38 - Reading TSK.csv (detected format: newline)
2025-04-27 17:12:38 - Loaded 72 player IDs from TSK.csv
2025-04-27 17:12:39 - Processing ༒ᶦᵃᵐKԋαʅҽҽʂι༒ (46068074) [1/72]
2025-04-27 17:12:40 - Found valid captcha with method: Eroded, confidence: 0.83
2025-04-27 17:12:40 - Recognized captcha: FXPA
2025-04-27 17:12:47 - Found valid captcha with method: Basic Threshold, confidence: 0.92
2025-04-27 17:12:47 - Recognized captcha: fY9U
2025-04-27 17:12:53 - Result: Already redeemed
2025-04-27 17:12:54 - Processing Mr Litosqui (44500803) [2/72]
2025-04-27 17:12:55 - Found valid captcha with method: Enhanced Contrast, confidence: 0.95
2025-04-27 17:12:55 - Recognized captcha: nsSz
2025-04-27 17:12:55 - Result: Already redeemed
... [Rest of the processing] ...
2025-04-29 19:32:14 - All 72/72 FIDs processed.
2025-04-29 19:32:14 - 
=== Redemption Complete ===
2025-04-29 19:32:14 - Successfully redeemed: 18
2025-04-29 19:32:14 - Already redeemed: 53
2025-04-29 19:32:14 - Errors/Failures: 1

=== Error Details ===
2025-04-29 19:32:14 - The following IDs encountered errors:
2025-04-29 19:32:14 - FID 138795240: Session expired after captcha

=== Captcha Statistics ===
2025-04-29 19:32:14 - Total attempts: 94
2025-04-29 19:32:14 - Successful decodes: 81
2025-04-29 19:32:14 - First attempt success: 62
2025-04-29 19:32:14 - Retries: 19
2025-04-29 19:32:14 - Complete failures: 0
2025-04-29 19:32:14 - Success rate: 86.17%
2025-04-29 19:32:14 - First try success rate: 76.54%
2025-04-29 19:32:14 - Average attempts per successful captcha: 1.16
2025-04-29 19:32:14 - Total execution time: 0:17:36
~~~

### Logging

All console output is automatically appended to a log file named `redeemed_codes.txt`, created in the same directory where the script is run.

If the `--all-images` flag is used, a folder named `failed_captchas` will be created in the script's directory. This folder will contain images of the CAPTCHAs the script attempted to solve, along with the OCR result in the filename (e.g., `fid12345678_try0_OCR_ABCD_timestamp.png`). This is useful for debugging CAPTCHA recognition issues.

---

## Troubleshooting

1.  **`pip` command not found**: Ensure Python is installed correctly and its `Scripts` directory (Windows) or equivalent (macOS/Linux) is in your system's PATH environment variable.
2.  **ModuleNotFoundError**: Make sure you've installed *all* the required libraries (`pip install requests easyocr opencv-python numpy torch torchvision torchaudio pillow`). Pay attention to any errors during installation.
3.  **Dependency Installation Issues**: `easyocr`, PyTorch, or `opencv-python` can sometimes be tricky to install. Consult their official documentation or search for OS-specific installation guides (e.g., installing C++ build tools might be needed). Ensure you have a compatible Python version.
4.  **CSV File Not Found**: Verify the path provided with `--csv` is correct. Check for typos. If using `*.csv` or a folder path, ensure `.csv` files actually exist there.
5.  **Permission Denied (Log File / CAPTCHA Folder)**: Ensure the script has permission to write files and create directories in the location it's running from. Try running the terminal/command prompt as an administrator (use with caution).
6.  **Code Expired / Claim Limit Reached**: The script will detect these specific API responses (`TIME ERROR`, `USED`) and stop execution, logging the reason. Verify the gift code's validity.
7.  **API Rate Limiting / `TIMEOUT RETRY` failures**: If you see many `Server requested retry` messages followed by failures, the API might be temporarily overloaded or you might be sending requests too quickly. Try increasing the `DELAY` value (in seconds) near the top of the script file (e.g., `DELAY = 2`).
8.  **CAPTCHA Solving Failures (`CAPTCHA CHECK ERROR`, `Sign Error`)**: If the script frequently fails with CAPTCHA errors, even after retries:
    *   The API's CAPTCHA might have changed its style (font, distortion, length), making it harder for EasyOCR to read.
    *   EasyOCR performance can vary. Ensure it's installed correctly. Using a GPU (if available and configured correctly with PyTorch) can improve speed and sometimes accuracy.
    *   If `CAPTCHA CHECK TOO FREQUENT` occurs often, the script will automatically add the player ID to a retry queue, continuing with other players.
9.  **EasyOCR Model Downloads**: The first time you run the script (or `easyocr`), it may need to download language models. Ensure you have an internet connection.
10. **GPU Issues**: If you get errors when using `--use-gpu`:
    * **For NVIDIA GPUs**:
      - Check your CUDA installation with the verification script in the GPU Support section
      - Make sure your PyTorch version matches your CUDA version
      - For most users, installing just the NVIDIA drivers and the PyTorch CUDA package is sufficient
    * **For AMD GPUs**:
      - Make sure ROCm is properly installed and your GPU is supported
      - AMD support is experimental and may not work with all models
    * Try specifying a different device ID (`--use-gpu 0` or `--use-gpu 1`)
    * If problems persist, run without the `--use-gpu` flag to use CPU mode

---

## Changelog

### v2.4.5 (Current Version)
- Added support for discrete GPUs with device selection via `--use-gpu` parameter
- Improved GPU error handling with graceful fallback to CPU

### v2.4.0
- Added improved CAPTCHA preprocessing with multiple image enhancement techniques
- Added comprehensive captcha statistics in summary output
- Added progress tracking with current/total indicators
- Added detailed error reporting by player ID
- Added execution time tracking and reporting
- Added final unprocessed ID check to guarantee all IDs are eventually processed
- Improved retry queue handling for rate-limited IDs
- Added option to disable captcha image saving to conserve disk space
- Fixed issues with counting and reporting

### v2.3.0
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