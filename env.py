import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools

class TradingEnv(gym.Env):
    """
    n stock trading enviornment.

    State Space: [# of shares owned, current stock prices, cash on hand]
    - array of length n_stock * 2 + 1
    - price is discretized to integer to reduce state space
    - using closing price of the stock
    - cash on hand is evaluated at each step based on action performace
    Possible Improvements:
    - Choose random starting points in the timeseries for every step

    Action Space: Sell (0), Hold (1), buy (2)
    - Sell all shares held when selling
    - Buy all shares that we can when buying
    - if multiple stocks 
    """

    def __init__(self, train_data, init_invest=20000):
        # Round data up to integer
        self.stock_price_history = np.around(train_data)
        self.n_stock, self.n_step = self.stock_price_history.shape

        self.init_invest = init_invest
        self.curr_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        # Action space
        self.action_space = spaces.Discrete(3**self.n_stock)

        # Observation space, give some estimates in order to sample
        stock_max_price = self.stock_price_history.max(axis=1)
        stock_range = [[0, init_invest * 2 // mx] for mx in stock_max_price] # range of the number of stocks we are able to purchase for each stock in the data
        price_range = [[0, mx] for mx in stock_max_price] # price ranges for the number of stocks in our bundle
        cash_in_hand_range = [[0, init_invest * 2]]
        self.observation_space = spaces.MultiDiscrete(stock_range + price_range + cash_in_hand_range)

        # seed and start
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # reset the enviornment 
    def _reset(self):
        self.curr_step = 0
        self.stock_owned = [0] * self.n_stock
        self.stock_price = self.stock_price_history[:, self.curr_step]
        self.cash_in_hand = self.init_invest
        return self._get_obs()

    def _get_val(self):
        return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand

    def _get_obs(self):
        obs = []
        obs.extend(self.stock_owned)
        obs.extend(list(self.stock_price))
        obs.append(self.cash_in_hand)
        return obs

    def _step(self, action):
        assert self.action_space.contains(action)
        prev_val = self._get_val()
        self.curr_step += 1
        self.stock_price = self.stock_price_history[:, self.curr_step] # updating price
        self._trade(action)
        curr_val = self._get_val()
        reward = curr_val - prev_val
        done = self.curr_step == self.n_step - 1
        info = {'curr_val': curr_val}
        return self._get_obs(), reward, done, info

    def _trade(self, action):
        # for each stock ticker, we can sell/hold/buy stocks
        action_combo = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
        action_vec = action_combo[action]

        # one pass to get sell/buy index
        sell_index = []
        buy_index = []
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        # sell first then buy
        if sell_index:
            for i in sell_index:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        
        if buy_index:
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1 # buy a share
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        can_buy = False




