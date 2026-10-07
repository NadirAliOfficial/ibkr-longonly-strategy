# IBKR Long-Only Strategy

## Description
A single‑file Python implementation of a long‑only trading strategy for Interactive Brokers (IBKR). It fetches daily price data directly from IBKR, computes entry signals using two technical indicators, applies a take‑profit rule based on a third indicator, and manages risk with a simple percentage stop‑loss. The script can be run in backtest mode to evaluate historical performance or in live mode for real‑time trading.

## Features
- **Entry signals** derived from two technical indicators (EMA and CMF).
- **Take‑profit** determined by a configurable indicator.
- **Stop‑loss** implemented as a fixed percentage of entry price.
- **Data source**: live market data directly from IBKR.
- **Modes**: backtest (using historical data) and live trading.
- **Configuration** via environment variables for IBKR connection details.

# IBKR Long-Only Strategy

Single-file automated trading strategy for **Interactive Brokers (IBKR)**.

- **Entries:** Based on 2 indicators  
- **Take-Profit:** From one indicator  
- **Stop-Loss:** Simple percentage  
- **Data:** Direct from IBKR only  
- **Modes:** Backtest + Live trading  

## Usage
```bash
python strategy.py --symbols AAPL --years 10 --exit-mode DAILY_EMA30 --sl-pct 0.02 --confirm-bars 2 --sl-arm-bars 2 --cooldown-bars 1
````

Configure IBKR host/port/client ID and stop-loss % inside `.env`.
<!-- updated: 2026-05-30 -->

## Installation
```bash
pip install -r requirements.txt
```

## Configuration
Create a `.env` file (or copy from `.env.example` if added) with the following variables:
```
IBKR_HOST=127.0.0.1        # IBKR gateway host
IBKR_PORT=7497             # IBKR gateway port
IBKR_CLIENT_ID=1          # Client ID for the IBKR API
```
These variables are read in `strategy.py` and `backtest.py` via `python-dotenv`.

## Project Structure
- `strategy.py` – entry point for live trading.
- `backtest.py` – script for running historical backtests.
- `requirements.txt` – Python dependencies.
- `README.md` – project documentation.
- `LICENSE` – MIT license.
- `.gitignore` – ignored files and directories.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
