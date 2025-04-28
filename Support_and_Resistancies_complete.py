# Import necessary libraries
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Add this import at the top of your code
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Import mdates for date formatting
from mplfinance.original_flavor import candlestick_ohlc
''''
# Download BTC data
def get_btc_data():
    btc = yf.Ticker("BTC-EUR")
    btc_data = btc.history(period="1y")
    return btc_data
'''
def get_data(ticker, period="1y", interval="1d"):
    """
    Fetch historical data for the given ticker.
    """
    try:
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
        # Flatten multi-level column names if they exist
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Calculate Jurik SMA (20)
def jurik_sma(series, period=20):
    return series.rolling(window=period).mean()

def round_to_multiple(value, multiple):
    """Round a value to the nearest multiple of a given parameter."""
    if multiple == 0 or multiple is None:
        raise ValueError("The 'multiple' parameter must be greater than 0.")
    return round(value / multiple) * multiple
# Funktion zur Berechnung der Trades

# Identify support and resistance levels
from scipy.signal import argrelextrema

def identify_support_resistance(df, past_window=10, future_window=3):
    """
    Identify support and resistance levels using local extrema with asymmetric windows.
    
    Parameters:
    - df: DataFrame containing the price data.
    - past_window: Number of days to look into the past for extrema.
    - future_window: Number of days to look into the future for extrema.
    
    Returns:
    - support: Series containing support levels.
    - resistance: Series containing resistance levels.
    """
    prices = df["Close"].values.ravel()

    # Identify local minima (support levels)
    local_min_idx = argrelextrema(prices, np.less, order=past_window + future_window)[0]
    support = pd.Series(prices[local_min_idx], index=df.index[local_min_idx])

    # Identify local maxima (resistance levels)
    local_max_idx = argrelextrema(prices, np.greater, order=past_window + future_window)[0]
    resistance = pd.Series(prices[local_max_idx], index=df.index[local_max_idx])

    return support, resistance

def long_trading_strategy(data, support, initial_capital=100000, share_rounding=0.01, future_window=3):
    """
    Execute the long trading strategy:
    - Buy at support levels after the future window delay.
    - Sell at resistance levels after the future window delay.
    """
    long_trades = []
    position = None  # Track current position: "long" or None
    total_long_profit = 0
    long_equity_curve = pd.Series(index=data.index, dtype=float)  # Initialize with data index
    current_equity = initial_capital

    for date, price in data["Close"].items():
        # Delay the buy by the future window
        if date in support.index and position is None:  # Only buy if no position is active
            current_index = support.index.get_loc(date)
            if current_index + future_window < len(support.index):  # Ensure index is within bounds
                delayed_date = support.index[current_index + future_window]
                shares_long = round_to_multiple(current_equity / data.loc[delayed_date, "Close"], share_rounding)
                long_trades.append((delayed_date, "BUY LONG", data.loc[delayed_date, "Close"], shares_long, 0))
                position = "long"
                position_price = data.loc[delayed_date, "Close"]

        # Delay the sell by the future window
        if date in resistance.index and position == "long":  # Only sell if a position is active
            current_index = resistance.index.get_loc(date)
            if current_index + future_window < len(resistance.index):  # Ensure index is within bounds
                delayed_date = resistance.index[current_index + future_window]
                shares_long = round_to_multiple(current_equity / position_price, share_rounding)
                profit = shares_long * (data.loc[delayed_date, "Close"] - position_price)
                total_long_profit += profit
                current_equity += profit
                long_equity_curve[delayed_date] = current_equity
                long_trades.append((delayed_date, "SELL LONG", data.loc[delayed_date, "Close"], shares_long, profit))
                position = None

    # Forward-fill equity curve to ensure all dates have values
    long_equity_curve.ffill(inplace=True)

    return long_trades, total_long_profit, long_equity_curve

def short_trading_strategy(data, resistance, initial_capital=100000, share_rounding=0.01, future_window=3):
    """
    Execute the short trading strategy:
    - Short at resistance levels after the future window delay.
    - Cover at support levels after the future window delay.
    """
    short_trades = []
    position = None  # Track current position: "short" or None
    total_short_profit = 0
    short_equity_curve = pd.Series(index=data.index, dtype=float)  # Initialize with data index
    current_equity = initial_capital

    for date, price in data["Close"].items():
        # Delay the short by the future window
        if date in resistance.index and position is None:  # Only short if no position is active
            current_index = resistance.index.get_loc(date)
            if current_index + future_window < len(resistance.index):  # Ensure index is within bounds
                delayed_date = resistance.index[current_index + future_window]
                shares_short = round_to_multiple(current_equity / data.loc[delayed_date, "Close"], share_rounding)
                short_trades.append((delayed_date, "SHORT", data.loc[delayed_date, "Close"], shares_short, 0))
                position = "short"
                position_price = data.loc[delayed_date, "Close"]

        # Delay the cover by the future window
        if date in support.index and position == "short":  # Only cover if a position is active
            current_index = support.index.get_loc(date)
            if current_index + future_window < len(support.index):  # Ensure index is within bounds
                delayed_date = support.index[current_index + future_window]
                shares_short = round_to_multiple(current_equity / position_price, share_rounding)
                profit = shares_short * (position_price - data.loc[delayed_date, "Close"])
                total_short_profit += profit
                current_equity += profit
                short_equity_curve[delayed_date] = current_equity
                short_trades.append((delayed_date, "COVER SHORT", data.loc[delayed_date, "Close"], shares_short, profit))
                position = None

    # Forward-fill equity curve to ensure all dates have values
    short_equity_curve.ffill(inplace=True)

    return short_trades, total_short_profit, short_equity_curve

def trading_strategy(data, support, resistance, initial_capital=100000, share_rounding=0.01):
    long_trades = []
    short_trades = []
    position = None  # Track current position: "long", "short", or None
    total_long_profit = 0
    total_short_profit = 0
    long_equity_curve = pd.Series(index=data.index, dtype=float)  # Initialize with data index
    short_equity_curve = pd.Series(index=data.index, dtype=float)  # Initialize with data index
    current_equity = initial_capital

    for date, price in data["Close"].items():
        # Check for support levels to open a long trade
        if date in support.index and position != "long":  # Open a long trade if no long position is active
            shares_long = round_to_multiple(current_equity / price, share_rounding)
            long_trades.append((date, "BUY LONG", price, shares_long, 0))  # Add PnL as 0 for opening trades
            print(f"Opened Long Trade: {long_trades[-1]}")  # Debug print
            position = "long"
            position_price = price

        # Check for resistance levels to open a short trade
        if date in resistance.index and position != "short":  # Open a short trade if no short position is active
            shares_short = round_to_multiple(current_equity / price, share_rounding)
            short_trades.append((date, "SHORT", price, shares_short, 0))  # Add PnL as 0 for opening trades
            print(f"Opened Short Trade: {short_trades[-1]}")  # Debug print
            position = "short"
            position_price = price

        # Close long position at resistance
        if date in resistance.index and position == "long":
            shares_long = round_to_multiple(current_equity / position_price, share_rounding)
            profit = shares_long * (price - position_price)
            total_long_profit += profit
            current_equity += profit
            long_equity_curve[date] = current_equity
            long_trades.append((date, "SELL LONG", price, shares_long, profit))  # Include PnL for closing trades
            print(f"Closed Long Trade: {long_trades[-1]}")  # Debug print
            position = None

        # Close short position at support
        if date in support.index and position == "short":
            shares_short = round_to_multiple(current_equity / position_price, share_rounding)
            profit = shares_short * (position_price - price)
            total_short_profit += profit
            current_equity += profit
            short_equity_curve[date] = current_equity
            short_trades.append((date, "COVER SHORT", price, shares_short, profit))  # Include PnL for closing trades
            print(f"Closed Short Trade: {short_trades[-1]}")  # Debug print
            position = None

    # Forward-fill equity curves to ensure all dates have values
    long_equity_curve.ffill(inplace=True)
    short_equity_curve.ffill(inplace=True)

    return long_trades, short_trades, total_long_profit, total_short_profit, long_equity_curve, short_equity_curve

# Calculate trade statistics

def calculate_trade_statistics(trades, equity_curve, initial_capital):
    if trades.empty:
        return {
            "Total Trades": 0,
            "Winning Trades": 0,
            "Losing Trades": 0,
            "Final Capital": initial_capital,
            "Win Percentage": 0,
            "Loss Percentage": 0,
            "Max Drawdown": 0,
            "Average Trade Return": 0
        }

    # Total trades
    total_trades = len(trades)

    # Winning and Losing trades
    winning_trades = trades[trades['PnL'] > 0].shape[0]
    losing_trades = trades[trades['PnL'] <= 0].shape[0]

    # Final capital from equity curve
    final_capital = equity_curve.iloc[-1]

    # Win and Loss percentages
    win_percentage = (winning_trades / total_trades) * 100
    loss_percentage = (losing_trades / total_trades) * 100

    # Average trade return
    avg_trade_return = trades['PnL'].mean()

    # Max Drawdown
    cumulative_pnl = equity_curve - initial_capital
    peak = cumulative_pnl.cummax()
    drawdown = (cumulative_pnl - peak) / peak
    max_drawdown = drawdown.min()

    return {
        "Total Trades": total_trades,
        "Winning Trades": winning_trades,
        "Losing Trades": losing_trades,
        "Final Capital": final_capital,
        "Win Percentage": win_percentage,
        "Loss Percentage": loss_percentage,
        "Max Drawdown": max_drawdown * -1,  # Convert to positive value
        "Average Trade Return": avg_trade_return
    }
def assign_signals(support, resistance, future_window=4):
    """
    Combine support and resistance levels and assign buy/sell signals.

    Parameters:
    - support: Series containing support levels.
    - resistance: Series containing resistance levels.
    - future_window: Number of days to delay the signals.

    Returns:
    - combined_df: DataFrame with combined levels and assigned signals.
    """
    # Combine support and resistance levels
    support_df = pd.DataFrame({'Date': support.index, 'Level': support.values, 'Type': 'support'})
    resistance_df = pd.DataFrame({'Date': resistance.index, 'Level': resistance.values, 'Type': 'resistance'})
    combined_df = pd.concat([support_df, resistance_df]).sort_values(by='Date').reset_index(drop=True)

    # Initialize columns for signals
    combined_df['Signal'] = None
    combined_df['Future Date'] = None

    # Assign signals based on type and future window (in days)
    for i, row in combined_df.iterrows():
        future_date = row['Date'] + pd.Timedelta(days=future_window)  # Add future_window days
        if row['Type'] == 'support':
            combined_df.at[i, 'Signal'] = 'buy'
            combined_df.at[i, 'Future Date'] = future_date
        elif row['Type'] == 'resistance':
            combined_df.at[i, 'Signal'] = 'sell'
            combined_df.at[i, 'Future Date'] = future_date

    return combined_df

def assign_long_trades(data, future_window=3):
    """
    Assign long trades based on support and resistance levels.

    Parameters:
    - data: DataFrame containing sorted support and resistance levels.
    - future_window: Number of days to delay the trades.

    Returns:
    - data: DataFrame with assigned long trades and future dates.
    """
    # Initialize columns for signals
    data['Long'] = None
    data['Long Date'] = None

    # Track the state of the long position
    long_active = False

    for i, row in data.iterrows():
        if row['Type'] == 'support' and not long_active:
            # Assign a buy signal
            data.at[i, 'Long'] = 'buy'
            data.at[i, 'Long Date'] = row['Date'] + pd.Timedelta(days=future_window)
            long_active = True
        elif row['Type'] == 'resistance' and long_active:
            # Assign a sell signal
            data.at[i, 'Long'] = 'sell'
            data.at[i, 'Long Date'] = row['Date'] + pd.Timedelta(days=future_window)
            long_active = False

    return data

def assign_short_trades(data, future_window=3):
    """
    Assign short trades based on resistance and support levels.

    Parameters:
    - data: DataFrame containing sorted support and resistance levels.
    - future_window: Number of days to delay the trades.

    Returns:
    - data: DataFrame with assigned short trades and future dates.
    """
    # Initialize columns for signals
    data['Short'] = None
    data['Short Date'] = None

    # Track the state of the short position
    short_active = False

    for i, row in data.iterrows():
        if row['Type'] == 'resistance' and not short_active:
            # Assign a short signal
            data.at[i, 'Short'] = 'short'
            data.at[i, 'Short Date'] = row['Date'] + pd.Timedelta(days=future_window)
            short_active = True
        elif row['Type'] == 'support' and short_active:
            # Assign a cover signal
            data.at[i, 'Short'] = 'cover'
            data.at[i, 'Short Date'] = row['Date'] + pd.Timedelta(days=future_window)
            short_active = False

    return data


def match_trades(data, support, resistance, future_window=3):
    long_trades = []
    short_trades = []
    position = None  # Track current position: "long", "short", or None

    # Iterate through the data index
    for i, date in enumerate(data.index):
        # Handle long trades
        if position is None and date in support.index:  # Buy long if no position is active
            current_index = support.index.get_loc(date)
            if current_index + future_window < len(data):  # Ensure index is within bounds
                delayed_date = data.index[current_index + future_window]
                shares = 1  # Placeholder for shares
                long_trades.append((delayed_date, "BUY LONG", data.loc[delayed_date, "Close"], shares))
                position = "long"

        if position == "long" and date in resistance.index:  # Sell long if a long position is active
            current_index = resistance.index.get_loc(date)
            if current_index + future_window < len(data):  # Ensure index is within bounds
                delayed_date = data.index[current_index + future_window]
                shares = 1  # Placeholder for shares
                long_trades.append((delayed_date, "SELL LONG", data.loc[delayed_date, "Close"], shares))
                short_trades.append((delayed_date, "SHORT", data.loc[delayed_date, "Close"], shares))
                position = "short"

        # Handle short trades
        if position is None and date in resistance.index:  # Open short if no position is active
            current_index = resistance.index.get_loc(date)
            if current_index + future_window < len(data):  # Ensure index is within bounds
                delayed_date = data.index[current_index + future_window]
                shares = 1  # Placeholder for shares
                short_trades.append((delayed_date, "SHORT", data.loc[delayed_date, "Close"], shares))
                position = "short"

        if position == "short" and date in support.index:  # Cover short if a short position is active
            current_index = support.index.get_loc(date)
            if current_index + future_window < len(data):  # Ensure index is within bounds
                delayed_date = data.index[current_index + future_window]
                shares = 1  # Placeholder for shares
                short_trades.append((delayed_date, "COVER SHORT", data.loc[delayed_date, "Close"], shares))
                position = None

    return long_trades, short_trades
    """
    Match trades for long and short strategies based on support and resistance levels.
    
    Rules:
    - Long trades:
        - Buy at the first support after a delay of `future_window` periods.
        - Sell at the first resistance after the support, delayed by `future_window` periods.
        - At the same time as the sell, open a short position.
    - Short trades:
        - Short at the first resistance after a delay of `future_window` periods.
        - Cover at the first support after the resistance, delayed by `future_window` periods.
    """
    long_trades = []
    short_trades = []
    position = None  # Track current position: "long", "short", or None

    # Iterate through the data index
    for i, date in enumerate(data.index):
        # Handle long trades
        if position is None and date in support.index:  # Buy long if no position is active
            current_index = support.index.get_loc(date)
            if current_index + future_window < len(data):  # Ensure index is within bounds
                delayed_date = data.index[current_index + future_window]
                long_trades.append((delayed_date, "BUY LONG", data.loc[delayed_date, "Close"]))
                position = "long"

        if position == "long" and date in resistance.index:  # Sell long if a long position is active
            current_index = resistance.index.get_loc(date)
            if current_index + future_window < len(data):  # Ensure index is within bounds
                delayed_date = data.index[current_index + future_window]
                long_trades.append((delayed_date, "SELL LONG", data.loc[delayed_date, "Close"]))
                short_trades.append((delayed_date, "SHORT", data.loc[delayed_date, "Close"]))
                position = "short"

        # Handle short trades
        if position is None and date in resistance.index:  # Open short if no position is active
            current_index = resistance.index.get_loc(date)
            if current_index + future_window < len(data):  # Ensure index is within bounds
                delayed_date = data.index[current_index + future_window]
                short_trades.append((delayed_date, "SHORT", data.loc[delayed_date, "Close"]))
                position = "short"

        if position == "short" and date in support.index:  # Cover short if a short position is active
            current_index = support.index.get_loc(date)
            if current_index + future_window < len(data):  # Ensure index is within bounds
                delayed_date = data.index[current_index + future_window]
                short_trades.append((delayed_date, "COVER SHORT", data.loc[delayed_date, "Close"]))
                position = None

    return long_trades, short_trades

def calculate_long_equity_curve(df, trades, initial_capital=10000):
    """
    Calculate the equity curve for long trades.
    The equity curve follows market moves only when invested.
    Includes open trades matched with the last close price.
    """
    equity_curve = []
    current_equity = initial_capital
    shares_long = 0

    for i in range(len(df)):
        row = df.iloc[i]

        # Check if today is the entry date of a long trade
        entry_trade = trades[(trades['Entry Date'] == row.name)]
        if not entry_trade.empty:
            shares_long = entry_trade.iloc[0]['Shares']
            equity_curve.append(current_equity)  # Equity remains the same on the entry day
            continue  # Skip calculation for the entry day

        # Update equity if a long position is active
        if shares_long > 0:
            current_equity += shares_long * (row['Close'] - df.iloc[i - 1]['Close'])

        # Check if today is the exit date of a long trade
        exit_trade = trades[(trades['Exit Date'] == row.name)]
        if not exit_trade.empty:
            shares_long = 0  # Reset shares after closing the position

        equity_curve.append(current_equity)

    # Handle open trades at the end of the dataset
    if shares_long > 0:
        last_close = df.iloc[-1]['Close']
        current_equity += shares_long * (last_close - df.iloc[-2]['Close'])
        equity_curve[-1] = current_equity  # Update the last equity value

    return pd.Series(equity_curve, index=df.index)

def calculate_short_equity_curve(df, trades, initial_capital=10000):
    """
    Calculate the equity curve for short trades.
    The equity curve follows market moves only when invested.
    Includes open trades matched with the last close price.
    """
    equity_curve = []
    current_equity = initial_capital
    shares_short = 0

    for i in range(len(df)):
        row = df.iloc[i]

        # Check if today is the entry date of a short trade
        entry_trade = trades[(trades['Entry Date'] == row.name)]
        if not entry_trade.empty:
            shares_short = entry_trade.iloc[0]['Shares']
            equity_curve.append(current_equity)  # Equity remains the same on the entry day
            continue  # Skip calculation for the entry day

        # Update equity if a short position is active
        if shares_short > 0:
            current_equity -= shares_short * (row['Close'] - df.iloc[i - 1]['Close'])

        # Check if today is the exit date of a short trade
        exit_trade = trades[(trades['Exit Date'] == row.name)]
        if not exit_trade.empty:
            shares_short = 0  # Reset shares after closing the position

        equity_curve.append(current_equity)

    # Handle open trades at the end of the dataset
    if shares_short > 0:
        last_close = df.iloc[-1]['Close']
        current_equity -= shares_short * (last_close - df.iloc[-2]['Close'])
        equity_curve[-1] = current_equity  # Update the last equity value

    return pd.Series(equity_curve, index=df.index)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

def plot_results(data, support, resistance, long_equity_curve, short_equity_curve):
    """
    Plot the results using Matplotlib, including:
    - Candlestick chart with support/resistance levels and Jurik SMA.
    - Equity curves for long and short trades.
    """
    # Prepare data for candlestick chart
    ohlc = data[['Open', 'High', 'Low', 'Close']].reset_index()
    ohlc['Date'] = mdates.date2num(ohlc['Date'])  # Convert dates to Matplotlib format
    ohlc = ohlc[['Date', 'Open', 'High', 'Low', 'Close']].values

    # Create the figure and subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])  # Candlestick chart is twice the height of equity curves

    # Plot the candlestick chart
    ax1 = fig.add_subplot(gs[0])
    candlestick_ohlc(ax1, ohlc, width=0.6, colorup='green', colordown='red', alpha=0.8)
    ax1.plot(data.index, data['Jurik_SMA_20'], color='orange', label='Jurik SMA (20)', linewidth=1.5)

    # Add support and resistance levels
    ax1.scatter(support.index, support, color='green', label='Support', s=20, zorder=5)
    ax1.scatter(resistance.index, resistance, color='blue', label='Resistance', s=20, zorder=5)

    # Format the candlestick chart
    ax1.set_title("BTC-EUR OHLC with Support/Resistance and Jurik SMA", fontsize=14)
    ax1.set_ylabel("Price (EUR)", fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.xaxis_date()  # Format x-axis as dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Plot the equity curves
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(long_equity_curve.index, long_equity_curve, label='Long Equity Curve', color='blue', linewidth=1.5)
    ax2.plot(short_equity_curve.index, short_equity_curve, label='Short Equity Curve', color='purple', linewidth=1.5)

    # Format the equity curves chart
    ax2.set_title("Equity Curves", fontsize=14)
    ax2.set_ylabel("Equity (EUR)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

    """
    Plot the results using Plotly, including:
    - Candlestick chart with support/resistance levels and Jurik SMA.
    - Equity curves for long and short trades.
    """
    fig = make_subplots(
        rows=2, cols=1,  # Merge the first two charts into one
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[2, 1],  # Increase the height of the candlestick chart (2x the equity curve)
        subplot_titles=("BTC-EUR OHLC with Support/Resistance and Jurik SMA", "Equity Curves")
    )

    # Add OHLC chart with green and red candles
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="BTC-EUR OHLC",
            increasing=dict(line=dict(color='green'), fillcolor='green'),  # Green for bullish candles
            decreasing=dict(line=dict(color='red'), fillcolor='red')      # Red for bearish candles
        ),
        row=1, col=1
    )

    # Add Jurik SMA line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Jurik_SMA_20'],
            mode='lines',
            name="Jurik SMA (20)",
            line=dict(color='orange', width=2)
        ),
        row=1, col=1
    )

    # Add support levels as green markers
    fig.add_trace(
        go.Scatter(
            x=support.index,
            y=support,
            mode='markers',
            name="Support",
            marker=dict(color='green', size=8)
        ),
        row=1, col=1
    )

    # Add resistance levels as red markers
    fig.add_trace(
        go.Scatter(
            x=resistance.index,
            y=resistance,
            mode='markers',
            name="Resistance",
            marker=dict(color='red', size=8)
        ),
        row=1, col=1
    )

    # Add long equity curve
    fig.add_trace(
        go.Scatter(
            x=long_equity_curve.index,
            y=long_equity_curve,
            name="Long Equity Curve",
            line=dict(color='blue')
        ),
        row=2, col=1
    )

    # Add short equity curve
    fig.add_trace(
        go.Scatter(
            x=short_equity_curve.index,
            y=short_equity_curve,
            name="Short Equity Curve",
            line=dict(color='purple')
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title="BTC-EUR Trading Strategy with Support/Resistance, Jurik SMA, and Equity Curves",
        height=900,  # Adjust overall height
        hovermode="x unified"
    )

    # Open the plot in the default browser
 #   fig.write_html("plot.html")
 #   import webbrowser
 #   webbrowser.open("plot.html")

import plotly.io as pio
pio.renderers.default = 'browser'
if __name__ == "__main__":
     # Get BTC data
    data = get_data("BTC-EUR", period="1y", interval="1d")

    # Ensure data is not None
    if data is None or data.empty:
        print("Error: No data fetched. Please check the ticker symbol or internet connection.")
        exit()

    # Calculate Jurik SMA (20)
    data['Jurik_SMA_20'] = jurik_sma(data['Close'], period=20)

    # Identify support and resistance levels
    support, resistance = identify_support_resistance(data, past_window=10, future_window=2)

    print("\nSupport Levels:")
    print(support)
    print("\nResistance Levels:")
    print(resistance)

    # Combine support and resistance into a single DataFrame
    support_df = pd.DataFrame({'Date': support.index, 'Level': support.values, 'Type': 'support'})
    resistance_df = pd.DataFrame({'Date': resistance.index, 'Level': resistance.values, 'Type': 'resistance'})
    combined_df = pd.concat([support_df, resistance_df]).sort_values(by='Date').reset_index(drop=True)

    # Assign long and short trades
    future_window = 3
    combined_df = assign_long_trades(combined_df, future_window=future_window)
    combined_df = assign_short_trades(combined_df, future_window=future_window)

    # Display the result
    print(combined_df)
    
    print("\nSupport Levels:")
    print(support)
    print("\nResistance Levels:")
    print(resistance)

    # Execute the long trading strategy with a future window delay
    long_trades, total_long_profit, long_equity_curve = long_trading_strategy(
        data, support, initial_capital=15000, share_rounding=0.01, future_window=3
    )

    # Execute the short trading strategy with a future window delay
    short_trades, total_short_profit, short_equity_curve = short_trading_strategy(
        data, resistance, initial_capital=15000, share_rounding=0.01, future_window=3
    )

    # Match trades based on support and resistance levels
    long_trades, short_trades = match_trades(data, support, resistance, future_window=3)
    print("\nLong Trades List:")
    print(long_trades)

    # Convert trades to DataFrames
    if long_trades:
        long_trades_df = pd.DataFrame(long_trades, columns=["Date", "Action", "Price", "Shares"])
    else:
        long_trades_df = pd.DataFrame(columns=["Date", "Action", "Price", "Shares"])

    if short_trades:
        short_trades_df = pd.DataFrame(short_trades, columns=["Date", "Action", "Price", "Shares"])
    else:
        short_trades_df = pd.DataFrame(columns=["Date", "Action", "Price", "Shares"])

    # Debug: Print the structure of the trades
    print("\nLong Trades DataFrame:")
    print(long_trades_df)
    print("\nShort Trades DataFrame:")
    print(short_trades_df)

    # Initialize PnL column for long trades
    if not long_trades_df.empty:
        long_trades_df['PnL'] = 0.0
        for i in range(1, len(long_trades_df), 2):  # Iterate over sell trades (every second row)
            buy_price = long_trades_df.loc[i - 1, 'Price']
            sell_price = long_trades_df.loc[i, 'Price']
            shares = long_trades_df.loc[i - 1, 'Shares']
            long_trades_df.loc[i, 'PnL'] = (sell_price - buy_price) * shares

    # Initialize PnL column for short trades
    if not short_trades_df.empty:
        short_trades_df['PnL'] = 0.0
        for i in range(1, len(short_trades_df), 2):  # Iterate over cover trades (every second row)
            short_price = short_trades_df.loc[i - 1, 'Price']
            cover_price = short_trades_df.loc[i, 'Price']
            shares = short_trades_df.loc[i - 1, 'Shares']
            short_trades_df.loc[i, 'PnL'] = (short_price - cover_price) * shares

    # Debug prints
    print("\nLong Trades DataFrame:")
    print(long_trades_df)
    print("\nShort Trades DataFrame:")
    print(short_trades_df)

    # Calculate and print trade statistics
    long_stats = calculate_trade_statistics(long_trades_df, long_equity_curve, initial_capital=15000)
    short_stats = calculate_trade_statistics(short_trades_df, short_equity_curve, initial_capital=15000)

    print("\nLong Trade Statistics:")
    for key, value in long_stats.items():
        print(f"{key}: {value}")

    print("\nShort Trade Statistics:")
    for key, value in short_stats.items():
        print(f"{key}: {value}")

    # Plot the results
    plot_results(data, support, resistance, long_equity_curve, short_equity_curve)
    
if __name__ == "__main__":
    # Define the future window parameter
    future_window = 4  # You can change this value as needed

    # Get BTC data
    data = get_data("BTC-EUR", period="1y", interval="1d")

    # Ensure data is not None
    if data is None or data.empty:
        print("Error: No data fetched. Please check the ticker symbol or internet connection.")
        exit()

    # Calculate Jurik SMA (20)
    data['Jurik_SMA_20'] = jurik_sma(data['Close'], period=20)

    # Identify support and resistance levels
    support, resistance = identify_support_resistance(data, past_window=10, future_window=future_window)

    print("\nSupport Levels:")
    print(support)
    print("\nResistance Levels:")
    print(resistance)

    # Execute the long trading strategy with the future window delay
    long_trades, total_long_profit, long_equity_curve = long_trading_strategy(
        data, support, initial_capital=100000, share_rounding=0.01, future_window=future_window
    )

    # Execute the short trading strategy with the future window delay
    short_trades, total_short_profit, short_equity_curve = short_trading_strategy(
        data, resistance, initial_capital=100000, share_rounding=0.01, future_window=future_window
    )

    # Match trades based on support and resistance levels
    long_trades, short_trades = match_trades(data, support, resistance, future_window=future_window)
    print("\nLong Trades List:")
    print(long_trades)
    # Combine support and resistance levels and assign signals
    combined_signals = assign_signals(support, resistance, future_window=future_window)
    print("\nCombined Signals:")
    print(combined_signals)

    # Convert trades to DataFrames
    if long_trades:
        long_trades_df = pd.DataFrame(long_trades, columns=["Date", "Action", "Price", "Shares"])
    else:
        long_trades_df = pd.DataFrame(columns=["Date", "Action", "Price", "Shares"])

    if short_trades:
        short_trades_df = pd.DataFrame(short_trades, columns=["Date", "Action", "Price", "Shares"])
    else:
        short_trades_df = pd.DataFrame(columns=["Date", "Action", "Price", "Shares"])

    # Debug: Print the structure of the trades
    print("\nLong Trades DataFrame:")
    print(long_trades_df)
    print("\nShort Trades DataFrame:")
    print(short_trades_df)

    # Initialize PnL column for long trades
    if not long_trades_df.empty:
        long_trades_df['PnL'] = 0.0
        for i in range(1, len(long_trades_df), 2):  # Iterate over sell trades (every second row)
            buy_price = long_trades_df.loc[i - 1, 'Price']
            sell_price = long_trades_df.loc[i, 'Price']
            shares = long_trades_df.loc[i - 1, 'Shares']
            long_trades_df.loc[i, 'PnL'] = (sell_price - buy_price) * shares

    # Initialize PnL column for short trades
    if not short_trades_df.empty:
        short_trades_df['PnL'] = 0.0
        for i in range(1, len(short_trades_df), 2):  # Iterate over cover trades (every second row)
            short_price = short_trades_df.loc[i - 1, 'Price']
            cover_price = short_trades_df.loc[i, 'Price']
            shares = short_trades_df.loc[i - 1, 'Shares']
            short_trades_df.loc[i, 'PnL'] = (short_price - cover_price) * shares

    # Calculate and print trade statistics
    long_stats = calculate_trade_statistics(long_trades_df, long_equity_curve, initial_capital=100000)
    short_stats = calculate_trade_statistics(short_trades_df, short_equity_curve, initial_capital=100000)

    print("\nLong Trade Statistics:")
    for key, value in long_stats.items():
        print(f"{key}: {value}")

    print("\nShort Trade Statistics:")
    for key, value in short_stats.items():
        print(f"{key}: {value}")

    # Plot the results
    plot_results(data, support, resistance, long_equity_curve, short_equity_curve)
