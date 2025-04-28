import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np


def flatten_dataframe(df):
    flattened_df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        flattened_df.columns = ['_'.join(col).strip() for col in df.columns.values]
    print(f"NaN values in flattened DataFrame: {flattened_df.isna().sum().sum()}")
    return flattened_df


def get_supertrend(high, low, close, period, multiplier):
    if len(close) < period:
        print(f"Not enough data points for period {period}.")
        return None, None, None

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    basic_upper = (high + low) / 2 + (multiplier * atr)
    basic_lower = (high + low) / 2 - (multiplier * atr)
    final_upper = pd.Series(0.0, index=close.index)
    final_lower = pd.Series(0.0, index=close.index)
    supertrend = pd.Series(0.0, index=close.index)

    for i in range(period, len(close)):
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
        if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]

    for i in range(period, len(close)):
        if supertrend.iloc[i-1] == final_upper.iloc[i-1] and close.iloc[i] <= final_upper.iloc[i]:
            supertrend.iloc[i] = final_upper.iloc[i]
        elif supertrend.iloc[i-1] == final_upper.iloc[i-1] and close.iloc[i] > final_upper.iloc[i]:
            supertrend.iloc[i] = final_lower.iloc[i]
        elif supertrend.iloc[i-1] == final_lower.iloc[i-1] and close.iloc[i] >= final_lower.iloc[i]:
            supertrend.iloc[i] = final_lower.iloc[i]
        elif supertrend.iloc[i-1] == final_lower.iloc[i-1] and close.iloc[i] < final_lower.iloc[i]:
            supertrend.iloc[i] = final_upper.iloc[i]
        else:
            supertrend.iloc[i] = 0.0

    combined = []
    colors = []
    Close = close.iloc[period:]

    for i in range(len(Close)):
        if Close.iloc[i] > supertrend.iloc[period + i]:
            combined.append(supertrend.iloc[period + i])
            colors.append('green')
        elif Close.iloc[i] < supertrend.iloc[period + i]:
            combined.append(supertrend.iloc[period + i])
            colors.append('red')
        else:
            combined.append(np.nan)
            colors.append('gray')

    st = pd.Series(supertrend.iloc[period:].values, index=Close.index)
    combined = pd.Series(combined, index=Close.index)
    colors = pd.Series(colors, index=Close.index)

    return st, combined, colors


def backtest_supertrend(stock_data, high_col, low_col, close_col, periods, multipliers, initial_capital=10000):
    best_params = None
    best_performance = -float('inf')
    results = []

    for period in periods:
        for multiplier in multipliers:
            st, combined_supertrend, supertrend_colors = get_supertrend(
                stock_data[high_col], stock_data[low_col], stock_data[close_col], period, multiplier
            )
            if st is None:
                continue

            stock_data['Supertrend'] = st
            total_return = (stock_data[close_col].iloc[-1] - stock_data[close_col].iloc[0]) / stock_data[close_col].iloc[0]
            results.append({
                'Period': period,
                'Multiplier': multiplier,
                'Total Return': total_return
            })

            if total_return > best_performance:
                best_performance = total_return
                best_params = (period, multiplier)

    results_df = pd.DataFrame(results)
    return best_params, results_df


def main():
    stock_symbol = input("Enter the stock symbol (e.g., BTC-USD, AAPL): ").strip()
    print(f"Running backtest for: {stock_symbol}")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Download data
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    if stock_data.empty:
        print("No data retrieved. Please check the stock symbol or date range.")
        return

    print("Data retrieved successfully:")
    print(stock_data.head())

    # Flatten and clean data
    stock_data = flatten_dataframe(stock_data)
    stock_data = stock_data.dropna()
    print("Data after cleaning:")
    print(stock_data.head())

    # Dynamically extract column names
    close_col = [col for col in stock_data.columns if "Close" in col][0]
    high_col = [col for col in stock_data.columns if "High" in col][0]
    low_col = [col for col in stock_data.columns if "Low" in col][0]

    print(f"Using columns: Close = {close_col}, High = {high_col}, Low = {low_col}")

    print("Optimizing Supertrend parameters...")
    periods = range(7, 15)
    multipliers = [1.5, 2.0, 2.5, 3.0]
    best_params, results_df = backtest_supertrend(stock_data, high_col, low_col, close_col, periods, multipliers)

    if best_params is None:
        print("No valid parameters found during optimization.")
        return

    print(f"Best Parameters: Period = {best_params[0]}, Multiplier = {best_params[1]}")
    print("Optimization Results:")
    print(results_df.sort_values(by='Total Return', ascending=False))

    # Calculate Supertrend using the best parameters
    st, combined_supertrend, supertrend_colors = get_supertrend(
        stock_data[high_col], stock_data[low_col], stock_data[close_col], best_params[0], best_params[1]
    )
    stock_data['Supertrend'] = st
    print(stock_data[[close_col, 'Supertrend']])


if __name__ == "__main__":
    main()