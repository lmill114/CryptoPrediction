# Cryptocurrency Prediction Code

This directory contains Jupyter Notebooks and Python files for cryptocurrency price prediction.

## CryptoPredicto.ipynb

The original cryptocurrency price prediction model using deep learning techniques.

## Technical Indicators and Enhanced Prediction Models

### Technical Indicators (technical_indicators.py)

A comprehensive set of technical indicators for cryptocurrency price analysis and prediction. It's designed to enhance machine learning models by providing engineered features that capture complex market dynamics.

#### Features
The `technical_indicators.py` script implements the following technical indicators:

1. **Moving Average Convergence Divergence (MACD)**
   - **Measures**: Trend direction, strength, momentum, and potential reversals
   - **Signals**: Trend changes, momentum shifts, and potential reversals
   - **Calculation**: Difference between 12-period and 26-period EMAs, with 9-period signal line

2. **Bollinger Bands**
   - **Measures**: Volatility, relative price levels, and potential overbought/oversold conditions
   - **Signals**: Volatility expansions/contractions and potential price reversals
   - **Calculation**: 20-period SMA with upper/lower bands at 2 standard deviations

3. **Stochastic Oscillator**
   - **Measures**: Momentum and relative position of closing price within a recent range
   - **Signals**: Overbought/oversold conditions and momentum shifts
   - **Calculation**: Compares current price to high-low range over 14 periods

4. **On-Balance Volume (OBV)**
   - **Measures**: Cumulative buying and selling pressure based on volume
   - **Signals**: Potential price breakouts and divergences
   - **Calculation**: Running total of volume, added when price rises, subtracted when price falls

5. **Average True Range (ATR)**
   - **Measures**: Market volatility
   - **Signals**: Potential for large price movements
   - **Calculation**: 14-period average of the true range (maximum of current high-low, high-previous close, previous close-low)

6. **Rate of Change (ROC)**
   - **Measures**: Price momentum as percentage change
   - **Signals**: Acceleration/deceleration in price movements
   - **Calculation**: Percentage change in price over 10 periods

7. **Commodity Channel Index (CCI)**
   - **Measures**: Cyclical overbought and oversold conditions
   - **Signals**: Potential trend reversals and cyclical turning points
   - **Calculation**: Deviation of typical price from its moving average, scaled by mean deviation

8. **Relative Strength Index (RSI)**
   - **Measures**: Momentum and overbought/oversold conditions
   - **Signals**: Potential price reversals and trend strength
   - **Calculation**: Ratio of average gains to average losses over 14 periods

9. **Ichimoku Cloud**
   - **Measures**: Support/resistance levels, trend direction, and momentum
   - **Signals**: Trend direction, momentum shifts, and future support/resistance
   - **Calculation**: Multiple components including Tenkan-sen, Kijun-sen, Senkou Span A/B, and Chikou Span

10. **Money Flow Index (MFI)**
    - **Measures**: Buying and selling pressure incorporating both price and volume
    - **Signals**: Overbought/oversold conditions and potential divergences
    - **Calculation**: Volume-weighted RSI over 14 periods

#### Usage

```python
import pandas as pd
import yfinance as yf
from technical_indicators import add_technical_indicators, plot_technical_indicators

# Download cryptocurrency data
symbol = "BTC-USD"
data = yf.download(symbol, start="2022-01-01", end="2023-01-01")

# Add technical indicators
data_with_indicators = add_technical_indicators(data)

# Print the first few rows
print(data_with_indicators.head())

# Plot the indicators
plot_technical_indicators(data_with_indicators, symbol)
```

#### Generating Trading Signals

The module can generate buy/sell signals based on each technical indicator:

```python
from technical_indicators import generate_trading_signals, plot_signals

# Generate trading signals
data_with_signals = generate_trading_signals(data_with_indicators)

# Plot the signals
plot_signals(data_with_signals, symbol)
```

### Enhanced CryptoPredicto (CryptoPredicto_with_indicators.py)

This script enhances the original CryptoPredicto cryptocurrency price prediction model by incorporating technical indicators and trading signals. It uses deep learning models (RNN, GRU, LSTM, and Transformer) to predict cryptocurrency prices with improved accuracy by leveraging technical analysis features.

#### Features

- **Technical Indicators Integration**: Incorporates 10+ technical indicators from the `technical_indicators.py` module
- **Trading Signals**: Uses buy/sell signals generated from each technical indicator
- **Multiple Deep Learning Models**:
  - Simple RNN with attention
  - Complex RNN with multiple layers
  - GRU with attention
  - LSTM with attention
  - Transformer with positional encoding
- **Comprehensive Evaluation**: Calculates MSE, RMSE, MAE, and R² metrics for model comparison
- **Visualization Tools**: Plots predictions, training losses, and feature importance

#### Usage

1. Ensure you have the required dependencies installed:
   ```
   pip install pandas numpy matplotlib torch scikit-learn
   ```

2. Make sure the `technical_indicators.py` file is in the same directory or in your Python path.

3. Prepare your cryptocurrency data files in CSV format with the following columns:
   - Date
   - Price (Close price)
   - Open
   - High
   - Low
   - Vol. (Volume)
   - Change %
   - Sentiment (if available)

4. Update the file paths in the script:
   ```python
   crypto_files = {
       'BTC': 'path/to/BitcoinData.csv',
       'ETH': 'path/to/EthereumData.csv',
       'DOGE': 'path/to/DogecoinData.csv'
   }
   ```

5. Run the script:
   ```
   python CryptoPredicto_with_indicators.py
   ```

### Additional Analysis Tools

#### Feature Importance Analysis (feature_importance.py)

This script provides tools for analyzing the importance of different features in cryptocurrency price prediction models. It uses various methods including:

- Random Forest feature importance
- Gradient Boosting feature importance
- Permutation feature importance
- SHAP (SHapley Additive exPlanations) values
- Feature correlation analysis

#### Market Cycle Detection (market_cycles.py)

This script provides tools for detecting market cycles, phases, and volatility regimes in cryptocurrency price data. Features include:

- Market phase detection using K-means clustering
- Market cycle detection by identifying local peaks and troughs
- Volatility regime detection using Hidden Markov Models
- Trend reversal detection
- Creation of market cycle-based features for prediction models

#### Temporal Feature Engineering (temporal_features.py)

This script provides tools for adding temporal features to cryptocurrency price data. Features include:

- Cyclical time features (hour, day, week, month, etc.)
- Holiday features
- Cryptocurrency-specific event features (e.g., Bitcoin halving)
- Seasonality features
- Autocorrelation features

## Benefits of Technical Indicators and Enhanced Models

The addition of technical indicators provides several benefits:

1. **Capture Market Patterns**: Technical indicators help identify patterns that raw price data alone might miss
2. **Reduce Overfitting**: Additional features can help the model generalize better
3. **Improve Timing**: Trading signals can help the model better time entry and exit points
4. **Handle Different Market Regimes**: Different indicators perform better in different market conditions

