import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta

# Function to calculate Jurik SMA
def jurik_sma(series, period=46):
    return series.rolling(window=period).mean()

# Get date range for last 365 days
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Download BTC data
btc = yf.Ticker("BTC-EUR")
data = btc.history(start=start_date, end=end_date)

# Calculate Jurik SMA
data['Jurik_SMA_20'] = jurik_sma(data['Close'], period=20)

# Initialize signals
data['Signal'] = 0  # 1 for Buy, -1 for Sell

# Generate Buy and Sell signals
for i in range(1, len(data)):
    if data['Close'].iloc[i] > data['Jurik_SMA_20'].iloc[i] and data['Close'].iloc[i-1] <= data['Jurik_SMA_20'].iloc[i-1]:
        data.loc[data.index[i], 'Signal'] = 1  # Buy signal
    elif data['Close'].iloc[i] < data['Jurik_SMA_20'].iloc[i] and data['Close'].iloc[i-1] >= data['Jurik_SMA_20'].iloc[i-1]:
        data.loc[data.index[i], 'Signal'] = -1  # Sell signal

# Create a trading list
def generate_trading_list(data):
    trades = []
    position = None
    entry_price = None
    entry_date = None

    for i in range(len(data)):
        row = data.iloc[i]
        if row['Signal'] == 1 and position is None:  # Buy signal
            position = 'Long'
            entry_price = row['Close']
            entry_date = row.name
        elif row['Signal'] == -1 and position == 'Long':  # Sell signal
            trades.append({
                'Entry Date': entry_date,
                'Exit Date': row.name,
                'Entry Price': entry_price,
                'Exit Price': row['Close'],
                'PnL': row['Close'] - entry_price,
                'Duration': (row.name - entry_date).days
            })
            position = None

    return pd.DataFrame(trades)

# Generate the trading list
trading_list = generate_trading_list(data)
print("\nTrading List:")
print(trading_list)

# Calculate trade statistics
def calculate_trade_statistics(trades):
    if trades.empty:
        return {
            "Total Profit": 0,
            "Average Profit": 0,
            "Number of Trades": 0,
            "Win Rate": 0,
            "Max Drawdown": 0
        }

    total_profit = trades['PnL'].sum()
    avg_profit = trades['PnL'].mean()
    num_trades = len(trades)
    win_rate = len(trades[trades['PnL'] > 0]) / num_trades * 100

    # Calculate max drawdown
    cumulative_pnl = trades['PnL'].cumsum()
    peak = cumulative_pnl.cummax()
    drawdown = (cumulative_pnl - peak) / peak
    max_drawdown = drawdown.min()

    return {
        "Total Profit": total_profit,
        "Average Profit": avg_profit,
        "Number of Trades": num_trades,
        "Win Rate": win_rate,
        "Max Drawdown": max_drawdown * -1  # Convert to positive value
    }

# Calculate statistics
trade_stats = calculate_trade_statistics(trading_list)
print("\nTrade Statistics:")
print(trade_stats)

# Generate equity curve
# Function to calculate equity curve based on daily close prices
# Function to calculate equity curve based on daily close prices
def calculate_long_equity_curve(df, trades, initial_capital=10000):
    equity_curve = []
    current_equity = initial_capital
    shares_long = 0

    for i in range(len(df)):
        row = df.iloc[i]

        # Check if today is the first day of a long trade
        entry_trade = trades[(trades['Entry Date'] == row.name)]
        if not entry_trade.empty:
            shares_long = 1  # Assume 1 share for simplicity
            equity_curve.append(current_equity)  # Equity remains the same on the entry day
            continue  # Skip calculation for the entry day

        # Update equity if a long position is active
        if shares_long > 0:
            current_equity += shares_long * (row['Close'] - df.iloc[i-1]['Close'])

        # Check if today is the last day of a long trade
        exit_trade = trades[(trades['Exit Date'] == row.name)]
        if not exit_trade.empty:
            shares_long = 0  # Reset shares after closing the position

        equity_curve.append(current_equity)

    return pd.Series(equity_curve, index=df.index)

# Calculate the equity curve and assign it to the DataFrame
data['Equity Curve'] = calculate_long_equity_curve(data, trading_list, initial_capital=100000)

# Print the equity curve
print(data[['Equity Curve']])

# Plot the equity curve
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data['Equity Curve'],
        mode='lines',
        name='Equity Curve',
        line=dict(color='blue')
    )
)

fig.update_layout(
    title='Equity Curve',
    xaxis_title='Time',
    yaxis_title='Equity',
    height=600
)

fig.show()