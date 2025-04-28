import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.trend import AroonIndicator, ADXIndicator
import plotly.io as pio

pio.renderers.default = 'browser'


def flatten_dataframe(df):
    flattened_df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        flattened_df.columns = ['_'.join(col).strip() for col in df.columns.values]
    print(f"NaN values in flattened DataFrame: {flattened_df.isna().sum().sum()}")
    return flattened_df


def get_supertrend(high, low, close, period, multiplier):
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


def calculate_aroon_adx(df, high_col, low_col, close_col):
    try:
        aroon_indicator = AroonIndicator(high=df[high_col], low=df[low_col], window=25)
        df[f'Aroon_Up_{close_col.split("_")[1]}'] = aroon_indicator.aroon_up()
        df[f'Aroon_Down_{close_col.split("_")[1]}'] = aroon_indicator.aroon_down()

        adx_indicator = ADXIndicator(high=df[high_col], low=df[low_col], close=df[close_col], window=14)
        df['ADX'] = adx_indicator.adx()
        df['ADX_Pos_DI'] = adx_indicator.adx_pos()
        df['ADX_Neg_DI'] = adx_indicator.adx_neg()

        print("Aroon and ADX calculation completed successfully.")
    except Exception as e:
        print(f"Error calculating Aroon and ADX: {e}")
    return df


class TradingSystem:
    def __init__(self, initial_capital=10000, position_size=2.0, stop_loss_pct=0.92, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.transaction_cost = transaction_cost

    def generate_trading_lists(self, df, symbol):
        long_trades = []
        short_trades = []
        long_current_position = None
        short_current_position = None
        long_entry_price = None
        short_entry_price = None

        for i in range(1, len(df)):
            close_price = df[f'Close_{symbol}'].iloc[i]
            supertrend = df['Supertrend'].iloc[i]
            previous_close_price = df[f'Close_{symbol}'].iloc[i-1]
            previous_supertrend = df['Supertrend'].iloc[i-1]

            if previous_supertrend > previous_close_price and supertrend < close_price:
                if short_current_position:
                    exit_price = close_price
                    profit_loss = (short_entry_price - exit_price) / short_entry_price
                    short_trades.append({
                        'entry_date': short_current_position['entry_date'],
                        'entry_price': short_entry_price,
                        'exit_date': df.index[i],
                        'exit_price': exit_price,
                        'profit_loss': profit_loss,
                        'type': 'Short'
                    })
                    short_current_position = None

                if not long_current_position:
                    long_current_position = {'entry_date': df.index[i]}
                    long_entry_price = close_price

            elif previous_supertrend < previous_close_price and supertrend >= close_price:
                if long_current_position:
                    exit_price = close_price
                    profit_loss = (exit_price - long_entry_price) / long_entry_price
                    long_trades.append({
                        'entry_date': long_current_position['entry_date'],
                        'entry_price': long_entry_price,
                        'exit_date': df.index[i],
                        'exit_price': exit_price,
                        'profit_loss': profit_loss,
                        'type': 'Long'
                    })
                    long_current_position = None

                if not short_current_position:
                    short_current_position = {'entry_date': df.index[i]}
                    short_entry_price = close_price

        return long_trades, short_trades


def main():
    stock_symbol = "BTC-EUR"
    system = TradingSystem()
    print(f"Running for symbol: {stock_symbol}")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    stock_data = flatten_dataframe(stock_data)

    close_col = f'Close_{stock_symbol}'
    high_col = f'High_{stock_symbol}'
    low_col = f'Low_{stock_symbol}'

    st, combined_supertrend, supertrend_colors = get_supertrend(
        stock_data[high_col], stock_data[low_col], stock_data[close_col], 7, 3
    )
    stock_data['Supertrend'] = st
    stock_data['CombinedSupertrend'] = combined_supertrend
    stock_data['SupertrendColors'] = supertrend_colors

    long_trades, short_trades = system.generate_trading_lists(stock_data, stock_symbol)

    stock_data = calculate_aroon_adx(stock_data, high_col, low_col, close_col)

    print("Long Trades:")
    print(pd.DataFrame(long_trades))
    print("\nShort Trades:")
    print(pd.DataFrame(short_trades))


if __name__ == "__main__":
    main()