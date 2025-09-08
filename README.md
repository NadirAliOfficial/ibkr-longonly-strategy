
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
