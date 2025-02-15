# Whiteout Survival Gift Code Redemption Script

Just a simple Python script that automates the process of redeeming gift codes in the game **Whiteout Survival**. It reads a list of Player IDs from a `.csv` file, sends requests to the game's giftcode redemption API, and redeems the specified gift code for each player.

---

## Features

- **CSV Import**: Reads player IDs from a `.csv` file. Each ID should be on a new line.
- **Command-Line Interface**: Accepts the `.csv` file path and gift code as arguments.
- **Error Handling**: Provides clear error messages for missing or invalid inputs.
- **Rate Limiting**: Adds a delay between requests to avoid triggering rate limits.

---

## Prerequisites

1. **Python 3.x**: Download and install Python from [python.org](https://www.python.org/).
2. **Required Libraries**: Install the required Python libraries using `pip`:
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

2. **Prepare the CSV File**:
   - Create a `.csv` file (e.g., `player_ids.csv`) with one player ID per row.
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

- `--csv`: Path to the `.csv` file containing player IDs.
- `--code`: The gift code to redeem.

### Example

```bash
python redeem_codes.py --csv player_ids.csv --code ILoveU
```

---

## Output

### Successful Run

```plaintext
Loaded 3 player IDs from player_ids.csv.
Processing Player ID: 57845354...
Result: SUCCESS
Processing Player ID: 98765432...
Result: RECEIVED.
Processing Player ID: 12345678...
Result: SUCCESS
```

---

## How It Works

1. **CSV Import**:
   - The script reads player IDs from the specified `.csv` file.

2. **API Requests**:
   - Sends a login request to validate the player ID.
   - Sends a redemption request to redeem the gift code.

3. **Sign Generation**:
   - Uses a secret key (`WOS_ENCRYPT_KEY`) to generate a `sign` parameter for each request.

4. **Rate Limiting**:
   - Adds a 1-second delay between requests to avoid being blocked.

---

## Troubleshooting

### Common Issues

1. **CSV File Not Found**:
   - Ensure the `.csv` file exists at the specified path.
   - Double-check the file name and extension.

2. **Invalid Gift Code**:
   - Verify that the gift code is correct and has not expired.

3. **API Rate Limiting**:
   - If the script is blocked, increase the delay between requests (`time.sleep(2)`).

---

## Future Enhancements

- **Logging**: Add logging to track successful and failed redemptions.
- **Retry Logic**: Implement retries for failed requests.
- **GUI**: Create a simple graphical interface for non-technical users.

---

## Changelog

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

If you encounter any issues or have questions, feel free to open an issue on the [GitHub repository](https://github.com/your-username/gift-code-redemption/issues).

---

## License

This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for details.
