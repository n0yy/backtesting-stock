import pandas as pd
from backtesting import Strategy, Backtest
from backtesting.lib import crossover

import talib

def MinMaxScaler(X, min, max):
    return (X - min) / (max - min)

def load_data(path:str):
    df = pd.read_csv(path, index_col="date", parse_dates=["date"])
    # drop some column
    df.drop(columns=["unix", "symbol", "Volume BTC"], inplace=True)
    # Rename Columns
    df.rename(columns={"open": "Open", 
               "close": "Close", 
               "low": "Low", 
               "high": "High", 
               "Volume USD": "Volume"}, inplace=True)
    # Scaling
    open_max, open_min = df.Open.max(), df.Open.min()
    close_max, close_min = df.Close.max(), df.Close.min()
    low_max, low_min = df.Low.max(), df.Low.min()
    high_max, high_min = df.High.max(), df.High.min()
    volume_max, volume_min = df.Volume.max(), df.Volume.min()

    df.Open = df.Open.apply(lambda x : MinMaxScaler(x, open_min, open_max))
    df.Close = df.Close.apply(lambda x : MinMaxScaler(x, close_min, close_max))
    df.Low = df.Low.apply(lambda x : MinMaxScaler(x, low_min, low_max))
    df.High = df.High.apply(lambda x : MinMaxScaler(x, high_min, high_max))
    df.Volume = df.Volume.apply(lambda x : MinMaxScaler(x, volume_min, volume_max))
    
    return df.loc["2022-01-01 00:00:00":]


def SMA(values, n):
    return pd.Series(values).rolling(n).mean()

class RSIOscillator(Strategy):
    upper_bound = 70
    lower_bound = 30
    
    def init(self):
        self.rsi = self.I(talib.RSI, self.data.Close, 14)

    def next(self):
        if crossover(self.lower_bound, self.rsi):
            self.buy()

        elif crossover(self.rsi, self.upper_bound):
            self.position.close()

    def get_last_rsi(self):
        return self.rsi[-1]


btc = load_data("data/btchourly.csv")

print(btc)

bt = Backtest(btc, RSIOscillator, cash=100)
stats = bt.optimize(lower_bound=range(10, 45, 5),
                    upper_bound=range(50, 95, 5),
                    maximize="Sharpe Ratio",
                    constraint=lambda param : param.lower_bound < param.upper_bound)

print(stats)
"""
last_rsi = stats._strategy.get_last_rsi()
print(last_rsi)
"""