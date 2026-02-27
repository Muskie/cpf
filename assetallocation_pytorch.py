#
# Based on
# Investing Environment and Agent
# Three Asset Case
#
# (c) Dr. Yves J. Hilpisch
# Reinforcement Learning for Finance
#

import os
import math
import random
import numpy as np
import pandas as pd
from scipy import stats
from pylab import plt, mpl
from scipy.optimize import minimize

import torch
from dqlagent_pytorch import *

plt.style.use('seaborn-v0_8')
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
np.set_printoptions(suppress=True)



class observation_space:
    def __init__(self, n):
        self.shape = (n,)


class action_space:
    def __init__(self, n):
        self.n = n
    def seed(self, seed):
        random.seed(seed)
    def sample(self):
        # This is also number of assets, it is a one dimensional array of n random numbers [0.0, 1.0)
        rn = np.random.random(self.n) 
        return rn / rn.sum()

class Investing:
    def __init__(self, 
                 path,
                 holdings,
                 steps=252, 
                 amount=1):
        self.path = path
        self.noa = len(holdings)
        self.holdings = holdings
        self.steps = steps
        self.initial_balance = amount
        self.portfolio_value = amount
        self.portfolio_value_new = amount
        self.observation_space = observation_space(self.noa + 1) # Number of assets plus one
        self.osn = self.observation_space.shape[0]
        self.action_space = action_space(self.noa) # Number of assets
        self.retrieved = 0
        self._generate_data()
        # This is the true payoff but we also need to keep holdings (tickers/symbols), their values, and weights too now
        self.asset_values = [] 
        self.asset_weights = []
        self.portfolios = pd.DataFrame()
        self.episode = 0

    def _generate_data(self):
        if self.retrieved:
            pass
        else:
            self.raw = pd.read_csv(self.path, index_col=0, parse_dates=True).dropna()
            self.retrieved # is this in nn.Module this must set it to 1 or at least non-zero
        self.data = self.raw[self.holdings]
        s = random.randint(self.steps, len(self.data))
        self.data = self.data.iloc[s-self.steps:s] # this starts in a random place
        # This divides by initial price, I think this is why he uses prices it sets the location zero as a one
        # This normalizes the returns, neural networks are sensitive to scale of data, it graphs nice too.
        self.data = self.data / self.data.iloc[0] 

    def _get_state(self):
        # This is a bit convoluted but it works
        self.asset_values = self.data[self.holdings].iloc[self.bar] 
        temp_array = self.asset_values.to_numpy() 
        combined = np.append(temp_array, self.asset_weights) 
        date = self.data.index[self.bar]
        our_array = np.array(combined), {'date': date}
        return our_array
        
    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            
    def reset(self):
        self.asset_weights = [0] * self.noa
        self.bar = 0 
        self.treward = 0
        self.portfolio_value = self.initial_balance
        self.portfolio_value_new = self.initial_balance
        self.episode += 1
        self._generate_data()
        self.state, info = self._get_state()
        return self.state, info

    def add_results(self, pl):
        # It was possible to make the dataframe more programatically
        # I did preserve the original design though we always address columns by name.
        # profit/loss (pl) is passed in, it can be zero so never divide by it.

        asset_weight_columns = []
        new_columns = []
        existing_states = []
        newest_states = []
        main_columns_list = []
        main_values_list = []
        counter = 0
        for symbol in self.holdings:
            weight_column = "weight_" + symbol
            asset_weight_columns.append(weight_column)
            new_column = symbol + "_new"
            new_columns.append(new_column)
            existing_states.append(self.state[counter])
            newest_states.append(self.new_state[counter])
            counter += 1

        # Apparently I want extend not append when I have multiple values
        main_columns_list.extend(['e', 'date'])
        main_columns_list.extend(asset_weight_columns)
        main_columns_list.extend(['pv', 'pv_new', 'p&l[$]', 'p&l[%]'])
        main_columns_list.extend(self.holdings)
        main_columns_list.extend(new_columns)

        # When it is a single value I can use append
        main_values_list.append(self.episode)
        main_values_list.append(self.date)
        main_values_list.extend(self.asset_weights)
        main_values_list.append(self.portfolio_value)
        main_values_list.append(self.portfolio_value_new)
        main_values_list.append(pl)
        percent_change = pl / self.portfolio_value_new * 100
        main_values_list.append(percent_change)
        main_values_list.extend(existing_states)
        main_values_list.extend(newest_states)

        data = dict(zip(main_columns_list, main_values_list))

        df = pd.DataFrame(data, index=[0])
        self.portfolios = pd.concat((self.portfolios, df), ignore_index=True)

    # If you get nan, you probably let one of the numbers below such as the new_pv go to infinity.
    def step(self, action):
        self.bar += 1
        self.new_state, info = self._get_state()
        self.date = info['date']
        if self.bar == 1:
            self.asset_weights = action
            pl = 0.
            reward = 0.
            self.add_results(pl)
        else:
            new_pv = 0
            counter = 0
            for symbol in self.holdings:
                new_pv += self.asset_weights[counter] * self.portfolio_value * self.new_state[counter] / self.state[counter]
                counter += 1
            self.portfolio_value_new = new_pv
            pl = self.portfolio_value_new - self.portfolio_value
            self.asset_weights = action
            self.add_results(pl)
            ret = self.portfolios['p&l[%]'].iloc[-1] / 100 * 252
            # Rolling 20 trading days is used here, because we have a random sample of at least 252 trading days
            vol = self.portfolios['p&l[%]'].rolling(
                20, min_periods=1).std().iloc[-1] * math.sqrt(252)
            sharpe = ret / vol
            reward = sharpe
            self.portfolio_value = self.portfolio_value_new
        if self.bar == len(self.data) - 1:
            done = True
        else:
            done = False
        self.state = self.new_state
        return self.state, reward, done, False, {}
        

class InvestingAgent(DQLAgent):
    def __init__(self, 
                 symbol, 
                 feature, 
                 n_features, 
                 env,
                 penalty_scaling_factor=1.0,
                 boundaries=None,
                 starting_weights=None,
                 hu=24, 
                 lr=0.001):
        super().__init__(symbol, feature, n_features, env, hu, lr)
        # I now pass in bnds and the option to use weights other than equal as our first guess.
        # A QNetwork is a Continuous action: override model to output scalar Q-value
        self.model = QNetwork(self.n_features, 1, hu).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr) # This is the optimizer used previously.
        self.criterion = nn.MSELoss() # criterion is Mean Square Error which we can reduce easier than RMSE.
        self.penalty_scaling_factor = penalty_scaling_factor # I expiremented with a bigger penalty.
        if ((boundaries is not None) and (len(boundaries) == self.env.noa)):
            self.bnds = boundaries
        else:
            self.bnds = self.env.noa * [(0, 1)]  # If none or bad boundaries use this
        if((starting_weights is not None) 
           and (len(starting_weights) == self.env.noa)
           and (math.isclose(sum(starting_weights), 1))):
            self.starting_weights = starting_weights
        else:
            self.starting_weights = self.env.noa * [1 / self.env.noa]
            
    def opt_action(self, state):
        cons = [{'type': 'eq', 'fun': lambda x: x.sum() - 1}]
        def f_obj(x):
            s = state.copy() # Must copy here and also account for more assets.
            s[0, self.env.noa:] = x 
            pen = np.mean((state[0, self.env.noa:] - x) ** 2) # Penalty term to every asset, we can scale this linearly.
            s_tensor = torch.FloatTensor(s).to(device)
            with torch.no_grad():
                q_val = self.model(s_tensor)
            return q_val.cpu().numpy()[0, 0] - (pen * self.penalty_scaling_factor) # Pay the (increased) penalty
        try:
            state = self._reshape(state)
            # SLSQP is Sequential Least Square Programing
            # eps is another epsilon step size, 1e-4 is considered big but that was what was used on the original dataset.
            # We are using minimize to maximize the q-value minus the penalty.
            res = minimize(lambda x: -f_obj(x), 
                           self.starting_weights,
                           bounds=self.bnds, 
                           constraints=cons,
                           options={'eps': 1e-4}, # You can set an option to limit the interations and potentially gain speed.  
                           method='SLSQP') 
            action = res['x'] # These are the weights that maximized.
        except Exception:
            # This catch just does a sample aka random action, previously he did the same action as before.
            action = self.env.action_space.sample()
        return action
        
    def act(self, state):
        if random.random() <= self.epsilon:
            our_action = self.env.action_space.sample()
        else:
            our_action = self.opt_action(state)
        return our_action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, next_state, reward, done in batch:
            target = torch.tensor([reward], dtype=torch.float32).to(device)
            if not done:
                ns = next_state.copy() # Again must copy
                action_cont = self.opt_action(ns)
                ns[0, self.env.noa:] = action_cont # Need to account for more assets to store
                ns_tensor = torch.FloatTensor(ns).to(device)
                with torch.no_grad():
                    future_q = self.model(ns_tensor)[0, 0]
                target = target + self.gamma * future_q
            state_tensor = torch.FloatTensor(state).to(device)
            self.optimizer.zero_grad()
            current_q = self.model(state_tensor)[0, 0]
            loss = self.criterion(current_q, target)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay # This is where epsilon decays between start and min

    def test(self, episodes, verbose=True):
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            state = self._reshape(state)
            treward = 0
            for _ in range(1, len(self.env.data) + 1):
                action = self.opt_action(state)
                # This is the key line, step returns state, reward (sharpe), done is True/False, trunc is always false and _
                # _ is an empty dictionary or set, it is always empty, probably for future use
                state, reward, done, trunc, _ = self.env.step(action)
                # print(_) # Always empty
                state = self._reshape(state)
                treward += reward
                if done:
                    templ = f'Episode {e} | Total Reward {treward:4.2f}'
                    if verbose:
                        print(templ, end='\r') # Also doesn't print so well on Google Colab
                    break
        print()

