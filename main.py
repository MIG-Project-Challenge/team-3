from algo import Algo
import argparse
import pandas as pd
import numpy as np
from mean_reversion import MeanReversion

class Trader:  # TODO: Implement Me
    def __init__(self, init_cash, stocks, algo):
        self.positions = [0 for i in range(len(stocks))]
        self.port_value = list()
        self.cash = init_cash
        self.total_short_value = 0
        self.data = list()  # Maybe a pandas dataframe?
        self.algo = algo()


def iterate_day(self, day_idx):
    self.algo.change_positions(self.positions, self.data, self.cash)
    self.check_valid(self.positions, self.cash, self.data)
    self.calc_port_value()
    self.print_data()


def check_valid(self):
    pass


def calc_port_value(self):
    pass


def print_value(self):
    pass


def eval_strategy(self, data):
    for idx, row in data.iterrows():
        self.data.append(row)
        self.iterate_day(idx)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Evaluate actions")
    parser.add_argument(
        "-a",
        "--actions",
        help="path to actions matrix file (should be .npy file)",
    )
    parser.add_argument(
        "-p",
        "--prices",
        help="path to stock prices matrix file (should be .npy file)",
    )

    args = parser.parse_args()

    prices = np.load(args.prices)
    actions = np.load(args.actions)

    assert prices.size() == actions.size(), "prices and actions must be the same size"
    trader = Trader(25000, 0, prices, MeanReversion)


