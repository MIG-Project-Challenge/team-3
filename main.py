## Loading the Data
import pandas as pd

# Read the training and testing data from CSV files
train_df = pd.read_csv('train_data_50.csv') # TODO: make sure that the train_data.csv file is in the same directory as this!

# change date into datetime objects
train_df['Date'] = pd.to_datetime(train_df['Date'])

# set indexes 
train_df.set_index(["Ticker", "Date"], inplace=True)

print(train_df)

# %%
# We can plot all the symbols like below
import matplotlib.pyplot as plt

tickers = sorted(train_df.index.get_level_values('Ticker').unique())

plt.figure(figsize=(8, 6))

for ticker in tickers:
    stock_close_data = train_df.loc[ticker]["Close"]
    plt.plot(stock_close_data.index,stock_close_data.values, label=f'{ticker}')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Prices')
plt.legend()
plt.show()

# %% [markdown]
# I will now create a Numpy array of all the stocks and their close prices. This will make it easier to manipulate later.

# %%
import numpy as np

tickers = sorted(train_df.index.get_level_values('Ticker').unique())

open_prices = []

for ticker in tickers:
    stock_close_data = train_df.loc[ticker]["Open"]
    open_prices.append(stock_close_data.values)

open_prices = np.stack(open_prices)
print(open_prices.shape)
print(open_prices)

# %%
close_prices = []

for ticker in tickers:
    stock_close_data = train_df.loc[ticker]["Close"]
    close_prices.append(stock_close_data.values)

close_prices = np.stack(close_prices)
print(close_prices.shape)
print(close_prices)

# %% [markdown]
# I will now make a trades Numpy array that is initialized to all zeros

# %%
trades = np.zeros_like(open_prices)
trades

# %% [markdown]
# Use talib to calculate technical indicators (in this case SMA).
# 
# The idea behind the strategy is that we want to have a slower moving average and faster moving average. When the faster SMA crosses above the slower SMA we buy, and when it crosses below the slower SMA we sell. The idea is that we are trying to buy when there is momentum and sell when the momentum slows.
# 
# Note here we are not keeping track of how much money we have spent and this could cause us to hit our cash limit or debt limit. We only have $25,000 dollars to spend so you will need to keep track of this as you spend it to buy shares. You will also need to keep track of your short positions value (because if it becomes negative and amounts to more than the value of your long positions and cash your portfolio will be liquidated and your algorithm will fail). 
# 
# You could take a look at the [eval_algo.py](eval_algo.py) script to see how we calculate your portfolios value.

# %%
# NOTE: that we don't use future price data to make trades in the past!
# This is to prevent "look-ahead bias" which is a common mistake in algorithmic trading, and will result in a score of 0 in the competition
# TODO: have a way to track your positions (short/long positions), your cash balance, adn your portfolio value
# recall the restrictions that you cannot buy a share if you don't have enough cash
# and you can't have your debt exceed the value of your cash and long positions
# "debt" in our case is really just the negative value of our short positions

#first step
period = 14

avgGain, avgLoss = 0, 0

for day in range(period):
    for stock in range(len(open_prices)):
        change = (close_prices[stock][day] - open_prices[stock][day])/open_prices[stock][day]
        if (change > 0):
            avgGain += change
        else:
            avgLoss += (-1 * change)

rsiVal = 100 - 100/(1 + avgGain/avgLoss)

moneyspent = 0

freq = np.zeros(len(open_prices))

for day in range(period, len(open_prices[0])-1):

    for stock in range(len(open_prices)): 

        change = (close_prices[stock][day] - open_prices[stock][day])/open_prices[stock][day]
    
        avgGain = avgGain * (period - 1)/period + max(0, change)/period
        avgLoss = avgLoss * (period - 1)/period - min(0, change)/period

        rsiVal = 100 - 100/(1 + avgGain/avgLoss)    
        
        k1, k2 = 3, 2
        
        if ((rsiVal < 30) and (moneyspent + k1 * open_prices[stock][day + 1] < 25000)):
            trades[stock][day + 1] = k1
            moneyspent += k1 * open_prices[stock][day + 1]
            freq[stock] += k1
        if ((rsiVal > 70) and (freq[stock] > k2 - 1)):
            trades[stock][day + 1] = (-1 * k2)
            moneyspent -= k2 * open_prices[stock][day + 1]
            freq[stock] -= k2


# %% [markdown]
# Now lets calculate the algorithm's returns:

# %%
from eval_algo import eval_actions

portfolio_value, sharpe_ratio = eval_actions(trades, open_prices, cash=25000, verbose=True)
print(f"\nPortfolio value: {portfolio_value}")
print(f"Sharpe ratio: {sharpe_ratio}")