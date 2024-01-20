import pandas as pd
from backtesting import Strategy, Backtest
from backtesting.lib import crossover

import talib

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
    
    return df


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


btc_daily = load_data("data/btcdaily.csv")

bt = Backtest(btc_daily, RSIOscillator, cash=1_000)
stats = bt.optimize(lower_bound=range(10, 40, 5),
                    upper_bound=range(50, 95, 5),
                    maximize="Sharpe Ratio",
                    constraint=lambda param : param.lower_bound < param.upper_bound)

print(stats)
print(stats._strategy)
print(stats._trades)