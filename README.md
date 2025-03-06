# Whiteout Survival Gift Code Redemption Script

Python script that automates the process of redeeming gift codes in the game **Whiteout Survival**. It reads a list of Player IDs from one or more `.csv` files, sends requests to the game's giftcode redemption API, and redeems the specified gift code for each player.

---

## Features

- **CSV Import**: Reads player IDs from a `.csv` file. Each ID should be on a new line.
- **Command-Line Interface**: Accepts the `.csv` file path (or `*.csv`) and gift code as arguments.
- **Error Handling**: Provides clear error messages for missing or invalid inputs.
- **Retry Logic**: Automatically retries failed requests when the server is busy.
- **Verbose Logging**: Shows timestamp, player nickname and ID during processing and logs them to a file.
- **Summary Report**: Provides a summary of successful redemptions, already redeemed codes, and errors at the end.
- **Rate Limiting**: Adds a delay between requests to avoid triggering rate limits.

---

## Prerequisites

1. **Python 3.x**: Download and install Python from [python.org](https://www.python.org/).
2. **Required Libraries**: Install the required Python libraries using `pip` from the command line:
   ```bash
   pip install requests
   ```

---

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/justncodes/wos-giftcode.git
   cd wos-giftcode
   ```
   ...or just [download the script](https://github.com/justncodes/wos-giftcode/blob/main/redeem_codes.py) directly.

2. **Prepare the CSV File**:
   - Create a `.csv` file (e.g., `player_ids.csv`) with one player ID per row.
   - Put the `.csv` file into the same folder as the script to avoid entering a path.
   - Example `player_ids.csv`:
     ```csv
     57845354
     98765432
     12345678
     ```

---

## Usage

Run the script from the command line with the following arguments:

```bash
python redeem_codes.py --csv <path_to_csv> --code <gift_code>
```

### Arguments

- `--csv`: Path to the `.csv` file containing player IDs, or `*.csv` to process all `.csv` files in the script directory.
- `--code`: The gift code to redeem.

### Example

Process a specific `.csv` file
```bash
python redeem_codes.py --csv player_ids.csv --code ILoveU
```

Process all `.csv` files in a specific folder
```bash
python redeem_codes.py --csv /path/to/folder --code ILoveU
```

Process all `.csv` files in the script's directory
```bash
python redeem_codes.py --csv *.csv --code ILoveU
```

---

## Output

### Sample Run

```plaintext
2025-02-20 22:00:00 - === Starting redemption for gift code: ILoveU at 2025-02-20 22:00:00 ===
2025-02-20 22:00:00 - Loaded 3 player IDs from player_ids.csv
2025-02-20 22:00:00 - Processing Cerberus (90321114)
2025-02-20 22:00:00 - Attempt 1: Server requested retry (TIMEOUT RETRY)
2025-02-20 22:00:02 - Result: Successfully redeemed
2025-02-20 22:00:03 - Processing Unknown Player (98765432)
2025-02-20 22:00:03 - Result: Already redeemed
2025-02-20 22:00:04 - Processing Amy (12345678)
2025-02-20 22:00:04 - Result: Error: Redemption failed

=== Redemption Summary ===
Successfully redeemed: 1
Already redeemed: 1
Errors: 1
```

### Logging

The script will append all output to a log file called redeemed_codes.txt in the same folder as the script.

---

## Troubleshooting

### Common Issues

1. **CSV File Not Found**:
   - Ensure the `.csv` file exists at the specified path.
   - Double-check the file name and extension.

2. **Invalid Gift Code**:
   - Verify that the gift code is correct and has not expired.

3. **API Rate Limiting**:
   - If the script is blocked too often, try increasing the DELAY between requests (line 16).

---

## Future Enhancements

- **GUI**: Create a simple graphical interface for non-technical users.

---

## Changelog

### v2.0.0 (Current Version)
- Added retry logic for failed requests (e.g., `TIMEOUT RETRY`).
- Display player nicknames during processing.
- Added Unicode support for special characters and foreign alphabets.
- Added a summary report of redemptions at the end.
- Improved error handling and logging.

### v1.0.0 (Initial Release)
- Added support for CSV import and command-line arguments.
- Implemented API request logic with sign generation.
- Added error handling and rate limiting.

---

## Credits

- **Author**: justncodes (\[SIR\] Yolo on #340)
- **Repository**: [wos-giftcode](https://github.com/justncodes/wos-giftcode)

---

## Support

If you encounter any issues or have questions, feel free to open an issue on the [GitHub repository](https://github.com/justncodes/wos-giftcode/issues).

---

## License

This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for details.
