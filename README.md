# Black-Scholes Options Calculator

A Python-based American options pricing calculator using the Cox-Ross-Rubinstein (CRR) binomial model. This tool fetches real-time market data and calculates fair values for options across all available expiration dates.

## Features

- **American Options Pricing**: Uses the CRR binomial model for accurate American options valuation
- **Real-time Data**: Fetches current stock prices and options data from Yahoo Finance
- **Dynamic Risk-free Rate**: Automatically selects appropriate Treasury rates based on time to expiration
- **Multiple Expirations**: Calculates fair values for all available option expiration dates
- **CSV Export**: Saves results to CSV files for further analysis
- **Flexible Parameters**: Customizable risk-free rate, dividend yield, and binomial steps

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Dependencies

Install the required packages:

```bash
pip install yfinance pandas numpy
```

## Usage

### Basic Usage

Run with default parameters (SPY ticker):

```bash
python main.py
```

### Advanced Usage

```bash
python main.py [TICKER] [OPTIONS]
```

#### Arguments

- `TICKER` (optional): Stock ticker symbol (default: SPY)

#### Options

- `--r RATE`: Risk-free rate (annualized, decimal). If omitted, auto-fetches per expiry
- `--q YIELD`: Dividend yield (annualized, decimal, default: 0.0)
- `--steps STEPS`: Number of binomial steps (default: 400)
- `--csv PATH`: Custom CSV output path

### Examples

```bash
# Price SPY options with default settings
python main.py

# Price AAPL options with custom risk-free rate
python main.py AAPL --r 0.045

# Price TSLA options with dividend yield and custom steps
python main.py TSLA --q 0.02 --steps 500

# Save to custom CSV file
python main.py MSFT --csv microsoft_options.csv
```

## Algorithm Details

### Binomial Model (Cox-Ross-Rubinstein)

The calculator uses the CRR binomial model to price American options, which allows for early exercise. Key parameters:

- **Up factor (u)**: `exp(σ√Δt)`
- **Down factor (d)**: `1/u`
- **Risk-neutral probability (p)**: `(exp((r-q)Δt) - d) / (u - d)`
- **Discount factor**: `exp(-r·Δt)`

### Risk-free Rate Selection

The tool automatically selects appropriate Treasury rates based on time to expiration:

- **≤ 13 weeks**: 3-Month Treasury (^IRX)
- **≤ 5 years**: 5-Year Treasury (^FVX)
- **> 5 years**: 10-Year Treasury (^TNX)

### Time to Expiration

Calculated as the number of days until expiration divided by 365, using UTC date comparison.

## Output Format

### Console Output

The tool displays:

- Current underlying price and parameters
- Options data grouped by expiration date
- Risk-free rates used for each expiration

### CSV Output

Contains columns:

- `expiry`: Option expiration date
- `type`: Option type (C for Call, P for Put)
- `strike`: Strike price
- `IV(%)`: Implied volatility percentage
- `mid`: Bid-ask midpoint (or last price if bid/ask unavailable)
- `fair_value`: Calculated fair value using binomial model
- `r_used`: Risk-free rate used in calculation

## File Structure

```
black-scholes/
├── main.py              # Main application
├── README.md            # This file
├── spy_08_16_2025.csv   # Example output file
└── open_08_16_2025.csv  # Example output file
```

## Key Functions

### `binomial_option()`

Core pricing function implementing the CRR binomial model for American options.

### `compute_fair_option_price()`

Main orchestration function that:

1. Fetches market data
2. Processes all expiration dates
3. Calculates fair values for all strikes
4. Formats and exports results

### `get_risk_free_rate_for_T()`

Automatically selects and fetches appropriate Treasury rates based on time to expiration.

## Limitations

- Requires active internet connection for data fetching
- Limited to options available on Yahoo Finance
- Assumes constant volatility (uses implied volatility from market)
- Uses simplified dividend modeling (constant yield)

## Error Handling

The tool includes robust error handling for:

- Invalid tickers
- Missing options data
- Network connectivity issues
- Data quality problems

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source, free to copy, or use as you please. Please ensure compliance with Yahoo Finance's terms of service when using their data.

## Disclaimer

This tool is for educational and research purposes only. Options trading involves significant risk. The calculated fair values should not be used as the sole basis for trading decisions. Always consult with a qualified financial advisor before making investment decisions.
