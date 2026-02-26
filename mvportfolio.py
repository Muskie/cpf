import math
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from pylab import plt
from scipy.optimize import minimize
from prettytable import PrettyTable
from datetime import datetime
plt.style.use('seaborn-v0_8')
pd.set_option("display.precision", 5)
np.set_printoptions(suppress=True,
        formatter={'float': lambda x: f'{x:.4f}'})

import warnings
warnings.filterwarnings('ignore') # I get a RuntimeWarning: Mean of empty slice. etc

# In the future perhaps connect to a database, this opens up more dimension data.
# The key requirement in the future will be handling strategies that are not as old as others.
class MVPortfolio:
    def __init__(self, 
                 universe, 
                 primary_exchange='TSX',
                 holdings=None, 
                 bm_returns=None,
                 asset_class_weights=None,
                 min_largest_asset_class_position_size=0.0,
                 max_largest_asset_class_position_size=1.0,
                 comparison_weights_one=None,
                 portfolio_one_name='Portfolio 1',
                 comparison_weights_two=None,
                 portfolio_two_name='Portfolio 2',
                 comparison_weights_three=None,
                 portfolio_three_name='Portfolio 3',
                 comparison_weights_four=None,
                 portfolio_four_name='Portfolio 4',
                 comparison_weights_five=None,
                 portfolio_five_name='Portfolio 5',
                 boundaries=None,
                 constraints=None):
        # Some more checking of the passed in returns would be wise.
        self.universe=universe
        self.primary_exchange=primary_exchange
        if holdings is None:
            self.holdings = self.universe.columns[:]
        else:
            self.holdings = holdings
        self.our_returns=self.universe[self.holdings]
        # To do annual properly you need a calendar
        self._set_calendar_and_key_dates()

        if ((bm_returns is not None) 
            and (len(bm_returns) == len(self.our_returns))
            and (self.our_returns.index[0] == bm_returns.index[0])
            and (self.our_returns.index[-1] == bm_returns.index[-1])):
            # Even better would be to check every date is there but for graphing this is fine.
            self.bm_returns = bm_returns
        else:
            self.bm_returns = None
        self.noa = len(self.holdings)
        self.equal_weights = self.noa * [1 / self.noa]
        # The reason there are five instead of one of these is for demo-ability, 
        # I wanted each of my teammates to have a horse in the race, a dictionary could be used instead.
        if((comparison_weights_one is not None) 
           and (len(comparison_weights_one) == self.noa)
           and (math.isclose(sum(comparison_weights_one), 1))):
            self.comparison_weights_one = comparison_weights_one
            self.portfolio_one_name = portfolio_one_name
        else:
            self.comparison_weights_one = self.equal_weights
        if((comparison_weights_two is not None) 
           and (len(comparison_weights_two) == self.noa)
           and (math.isclose(sum(comparison_weights_two), 1))):
            self.comparison_weights_two = comparison_weights_two
            self.portfolio_two_name = portfolio_two_name
        else:
            self.comparison_weights_two = self.equal_weights
        if((comparison_weights_three is not None) 
           and (len(comparison_weights_three) == self.noa)
           and (math.isclose(sum(comparison_weights_three), 1))):
            self.comparison_weights_three = comparison_weights_three
            self.portfolio_three_name = portfolio_three_name
        else:
            self.comparison_weights_three = self.equal_weights
        if((comparison_weights_four is not None) 
           and (len(comparison_weights_four) == self.noa)
           and (math.isclose(sum(comparison_weights_four), 1))):
            self.comparison_weights_four = comparison_weights_four
            self.portfolio_four_name = portfolio_four_name
        else:
            self.comparison_weights_four = self.equal_weights
        if((comparison_weights_five is not None) 
           and (len(comparison_weights_five) == self.noa)
           and (math.isclose(sum(comparison_weights_five), 1))):
            self.comparison_weights_five = comparison_weights_five
            self.portfolio_five_name = portfolio_five_name
        else:
            self.comparison_weights_five = self.equal_weights  
        self._determine_asset_classes(asset_class_weights)
        self._set_boundaries_and_constraints(min_largest_asset_class_position_size,
                                             max_largest_asset_class_position_size,
                                             boundaries,
                                             constraints)

    def _set_calendar_and_key_dates(self):
        self.calendar = mcal.get_calendar(self.primary_exchange)
        
        first_date_of_data = self.our_returns.index[0]
        first_year_of_data = first_date_of_data.year
        year_start_date = datetime(first_year_of_data, 1, 1)
        year_end_date = datetime(first_year_of_data, 12, 31)
        schedule = self.calendar.schedule(start_date=year_start_date, end_date=year_end_date)
        if not schedule.empty:
            self.first_annual_trading_date = schedule.index[0].to_pydatetime()
        # This will advance us one year
        if(self.first_annual_trading_date < first_date_of_data):
            next_year_of_data = first_year_of_data + 1
            year_start_date = datetime(next_year_of_data, 1, 1)
            year_end_date = datetime(next_year_of_data, 12, 31)
            schedule = self.calendar.schedule(start_date=year_start_date, end_date=year_end_date)
            self.first_annual_trading_date = schedule.index[0].to_pydatetime()
        # print(self.first_annual_trading_date)

        last_date_of_data = self.our_returns.index[-1]
        last_year_of_data = last_date_of_data.year
        year_start_date = datetime(last_year_of_data, 1, 1)
        year_end_date = datetime(last_year_of_data, 12, 31)
        schedule = self.calendar.schedule(start_date=year_start_date, end_date=year_end_date)
        if not schedule.empty:
            self.last_annual_trading_date = schedule.index[-1].to_pydatetime()
        # This will reduce one year but perhaps isn't perfect, but it is safe
        if(self.last_annual_trading_date.month > last_date_of_data.month):
            prev_year_of_data = last_year_of_data - 1
            year_start_date = datetime(prev_year_of_data, 1, 1)
            year_end_date = datetime(prev_year_of_data, 12, 31)
            schedule = self.calendar.schedule(start_date=year_start_date, end_date=year_end_date)
            self.last_annual_trading_date = schedule.index[-1].to_pydatetime()
        # print(self.last_annual_trading_date)

        
    def _determine_asset_classes(self,
                                 asset_class_weights=None):
        # This can be overriden by specifying the boundaries explicitly or other logic
        self.equity_holdings = []
        self.debt_holdings = []
        self.alternative_holdings = []
        for holding in self.holdings:
            asset_class = holding[:2]
            if (asset_class == 'E-'):
                self.equity_holdings.append(holding)
            elif (asset_class == 'D-'):
                self.debt_holdings.append(holding)
            else:
                self.alternative_holdings.append(holding)
                # The default is alternative, so if no prefix everything is in one asset class.
            
        if asset_class_weights is None:
            self.asset_class_weights = [0.6, 0.4, 0.0] 
        elif(math.isclose(sum(asset_class_weights), 1)):
            # We need to NOT have a weight in an asset class if we have zero holdings in that asset class.
            if(len(self.alternative_holdings) < asset_class_weights[2]):
               if(len(self.debt_holdings) < asset_class_weights[1]):
                   # All equity
                   self.asset_class_weights = [1, 0, 0]
               else:
                   self.asset_class_weights = [0.6, 0.4, 0] # This is the default 60:40 portfolio
            elif(len(self.debt_holdings) < asset_class_weights[1]):
                if(len(self.equity_holdings) < asset_class_weights[0]):
                    # All alternative
                    self.asset_class_weights = [0, 0, 1]
                else:
                    self.asset_class_weights = [0.6, 0, 0.4] # Variation on 60:40 portoflio
            elif(len(self.equity_holdings) < asset_class_weights[0]):
                # We have debt and alternative
                self.asset_class_weights = [0, 0.6, 0.4] # Variation on 60:40 portoflio
            else:
                # We have all the asset classes
                # print("We are here")
                self.asset_class_weights = asset_class_weights
        else:
            # print("We are actually here")
            self.asset_class_weights = asset_class_weights

    # Target asset class weights can be overriden by passing in exacting boundaries (bnds)
    def target_equity_weight(self):
        return self.asset_class_weights[0]

    def target_debt_weight(self):
        return self.asset_class_weights[1]

    def target_alt_weight(self):
        return self.asset_class_weights[2]

    def _set_boundaries_and_constraints(self,
                                        min_largest_asset_class_position_size=0.0,
                                        max_largest_asset_class_position_size=1.0,
                                        boundaries=None,
                                        constraints=None):
        # I let people override contraints and boundaries, constraints is simple, boundaries is convoluted.
        if constraints is None:
            self.cons = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}) # I was missing these brackets
        else:
            self.cons = constraints

        if ((boundaries is not None) and (len(boundaries) == self.noa)):
            self.bnds = boundaries
        else:
            # Asset class weights overrides min and max position size
            # To determine minimum I need to remember I have a budget of 1.00
            # The biggest asset class is the most important and must be the most accurately calculated and optimized
            # I don't let there be an asset class weight if there are no holdings in that asset class
            if ((self.target_equity_weight() > self.target_debt_weight()) 
                and (self.target_equity_weight() > self.target_alt_weight())
                and (self.target_alt_weight() > 0)
                and (self.target_debt_weight() > 0)):
                # This is the most likley scenario and no divide by zero
                min_alternative_position_size = self.target_alt_weight() / len(self.alternative_holdings)
                max_alternative_position_size = self.target_alt_weight()
                min_debt_position_size = self.target_debt_weight() / len(self.debt_holdings)
                max_debt_position_size = self.target_debt_weight()
                max_equity_position_size = max_largest_asset_class_position_size
                if((len(self.equity_holdings) * min_largest_asset_class_position_size) > self.target_equity_weight()):
                    # minimum is too big
                    min_equity_position_size = self.target_equity_weight() / len(self.equity_holdings)
                else:
                    min_equity_position_size = min_largest_asset_class_position_size
            elif((self.target_debt_weight() > self.target_equity_weight()) 
                 and (self.target_debt_weight() > self.target_alt_weight())
                 and (self.target_alt_weight() > 0)
                 and (self.target_equity_weight() > 0)):
                # This is the second most likely scenario and no divide by zero
                min_alternative_position_size = self.target_alt_weight() / len(self.alternative_holdings)
                max_alternative_position_size = self.target_alt_weight()
                min_equity_position_size = self.target_equity_weight() / len(self.equity_holdings)
                max_equity_position_size = self.target_equity_weight()
                max_debt_position_size = max_largest_asset_class_position_size
                if((len(self.debt_holdings) * min_largest_asset_class_position_size) > self.target_debt_weight()):
                    # minimum is too big
                    min_debt_position_size = self.target_debt_weight() / len(self.debt_holdings)
                else:
                    min_debt_position_size = min_largest_asset_class_position_size
            elif((self.target_alt_weight() > self.target_equity_weight()) 
                 and (self.target_alt_weight() > self.target_debt_weight())
                 and (self.target_debt_weight() > 0)
                 and (self.target_equity_weight() > 0)):
                # This is probably the next most likely scenario, no divide by zero
                min_debt_position_size = self.target_debt_weight() / len(self.debt_holdings)
                min_equity_position_size = self.target_equity_weight() / len(self.equity_holdings)
                max_debt_position_size = self.target_debt_weight()
                max_equity_position_size = self.target_equity_weight()
                max_alternative_position_size = max_largest_asset_class_position_size
                if((len(self.alternative_holdings) * min_largest_asset_class_position_size) > self.target_alt_weight()):
                    # minimum is too big
                    min_alternative_position_size = self.target_alt_weight() / len(self.alternative_holdings)
                else:
                    min_alternative_position_size = min_largest_asset_class_position_size
            else:
                # Convoluted logic to prevent bad boundaries
                if(((len(self.alternative_holdings) + len(self.debt_holdings) + len(self.equity_holdings)) 
                        * min_largest_asset_class_position_size) > 1):
                    # minimum is too big
                    if(len(self.alternative_holdings) > 0):
                        min_alternative_position_size = self.target_alt_weight() / len(self.alternative_holdings)
                        max_alternative_position_size = max_largest_asset_class_position_size
                    else:
                        min_alternative_position_size = 0
                        max_alternative_position_size = 0
                    if(len(self.debt_holdings) > 0):
                        min_debt_position_size = self.target_debt_weight() / len(self.debt_holdings)
                        max_debt_position_size = max_largest_asset_class_position_size
                    else:
                        min_debt_position_size = 0
                        max_debt_position_size = 0
                    if(len(self.equity_holdings) > 0):
                        min_equity_position_size = self.target_equity_weight() / len(self.equity_holdings)
                        max_equity_position_size = max_largest_asset_class_position_size
                    else:
                        min_equity_position_size = 0
                        max_equity_position_size = 0
                else:
                    # This is the simplest and ultimate fall back boundaries
                    min_alternative_position_size = 0
                    min_debt_position_size = 0
                    min_equity_position_size = 0
                    max_alternative_position_size = self.target_alt_weight()
                    max_debt_position_size = self.target_debt_weight()
                    max_equity_position_size = self.target_equity_weight()
               
            # I still need more overrides to have the cleanest looking boundaries (bnds)
            if(self.target_equity_weight() == 0):
                min_equity_position_size = 0
                max_equity_position_size = 0
            if(self.target_debt_weight() == 0):
                min_debt_position_size = 0
                max_debt_position_size = 0
            if(self.target_alt_weight() == 0):
                min_alternative_position_size = 0
                max_alternative_position_size = 0
    
            # Now we must step through the holdings and make our final list of bnds
            list_of_bounds = []
            for holding in self.holdings:
                asset_class = holding[:2]
                if (asset_class == 'E-'):
                    list_of_bounds.append(tuple((min_equity_position_size, max_equity_position_size)))
                elif (asset_class == 'D-'):
                    list_of_bounds.append(tuple((min_debt_position_size, max_debt_position_size)))
                else:
                    list_of_bounds.append(tuple((min_alternative_position_size, max_alternative_position_size)))
    
            self.bnds = tuple(list_of_bounds) 

        # I always want to know this as you really need to look at all the logic above carefully sometimes.
        print("Remember target asset class weights", self.asset_class_weights, "can be overruled by bnds.")
        print("Current bnds are", self.bnds)

    # This is my two static one class method solution
    @staticmethod        
    def annualized_return(rets, weights):
        return np.dot(rets.mean(), weights) * 252  

    @staticmethod
    def annualized_volatility(rets, weights):
        return math.sqrt(np.dot(weights, np.dot(rets.cov() * 252 , weights)))

    @classmethod
    def sharpe_ratio(cls, rets, weights):
        return cls.annualized_return(rets, weights) / cls.annualized_volatility(rets, weights)

    def maximum_return_portfolio(self):
        opt = minimize(lambda weights: -MVPortfolio.annualized_return(self.our_returns, weights),
                       self.equal_weights,
                       bounds=self.bnds,
                       constraints=self.cons)
        return opt['x']

    def minimum_risk_portfolio(self):
        opt = minimize(lambda weights: MVPortfolio.annualized_volatility(self.our_returns, weights),
                       self.equal_weights, 
                       bounds=self.bnds, 
                       constraints=self.cons)
        return opt['x']

    def maximum_sharpe_portfolio(self):
        opt = minimize(lambda weights: -MVPortfolio.sharpe_ratio(self.our_returns, weights),
                       self.equal_weights,
                       bounds=self.bnds,
                       constraints=self.cons)
        return opt['x']

    def pretty_weights(self, weights):
        table = PrettyTable()
        table.field_names = ['Holding', 'Weight']

        for holding in range(self.noa):
            table.add_row([self.holdings[holding], round(weights[holding],4)]) # I round to four digits for display
        print(table)

    def daily_portfolio_returns(self, weights, start_date=None, end_date=None):
        if(start_date is None):
            start_date = self.our_returns.index[0]
        if(end_date is None):
            end_date = self.our_returns.index[-1]
        return self.our_returns[start_date:end_date].mul(weights, axis=1).sum(axis=1)

    def cummulative_portfolio_returns(self, weights, start_date=None, end_date=None):
        if(start_date is None):
            start_date = self.our_returns.index[0]
        if(end_date is None):
            end_date = self.our_returns.index[-1]
        daily_portfolio_returns = self.daily_portfolio_returns(weights, start_date, end_date)
        return daily_portfolio_returns.cumsum().apply(np.exp)

    def annual_maximum_sharpe_portfolios(self):
        # I tried using pandas_market_calendars in the constructor but may need to get more gheto
        year_one = self.first_annual_trading_date.year
        last_year = self.last_annual_trading_date.year
        last_year_plus_one = last_year + 1
        optimal_annual_weights = {}
        for year in range(year_one, last_year_plus_one):
            rets_ = self.our_returns.loc[f'{year}-01-01':f'{year}-12-31']
            ow = minimize(lambda weights: -MVPortfolio.sharpe_ratio(rets_, weights),
                               self.equal_weights,
                               bounds=self.bnds,
                               constraints=self.cons)['x']
            optimal_annual_weights[year] = ow
        return optimal_annual_weights

    def pretty_annual_weights(self, annual_weights):
        last_year = self.last_annual_trading_date.year
        annual_weight_table = PrettyTable(["Year", "Holding", "Weights"])
        for key, val in annual_weights.items():
            for holding in range(self.noa):
                annual_weight_table.add_row([key, self.holdings[holding], val[holding].round(2)]) # round for display
            empty_row = [" "] * len(annual_weight_table.field_names)
            if(key != last_year): # This prevents the last empty row
                annual_weight_table.add_row(empty_row) 
        print(annual_weight_table)


    def visualize_efficient_frontier(self, show_true_max=True):
        extra_cons = ({'type': 'eq', 
                       'fun': lambda weights: MVPortfolio.annualized_return(self.our_returns, weights) - tret},
                      self.cons)
        # Draw a line from minimum_risk_weights to highest return possible within bounds
        minimum_risk_weights = self.minimum_risk_portfolio()
        minimum_risk_return = self.annualized_return(self.our_returns, minimum_risk_weights)
        minimum_risk_volatility = self.annualized_volatility(self.our_returns, minimum_risk_weights)
        maximum_sharpe_weights = self.maximum_sharpe_portfolio()
        maximum_sharpe_return = self.annualized_return(self.our_returns, maximum_sharpe_weights)
        maximum_sharpe_volatility = self.annualized_volatility(self.our_returns, maximum_sharpe_weights)
        equal_weight_return = self.annualized_return(self.our_returns, self.equal_weights)
        equal_weight_volatility = self.annualized_volatility(self.our_returns, self.equal_weights)

        # I declare these and overwrite them later if possible
        bm_annual_return = minimum_risk_return
        # For demoability I do this five times at least in my latest proof of concept.
        comparison_return_one = minimum_risk_return
        comparison_return_two = minimum_risk_return 
        comparison_return_three = minimum_risk_return 
        comparison_return_four = minimum_risk_return 
        comparison_return_five = minimum_risk_return 
        
        comparison_check_one = True
        comparison_check_two = True
        comparison_check_three = True
        comparison_check_four = True
        comparison_check_five = True
        if (len(self.equal_weights) != len(self.comparison_weights_one)):
            comparison_check_one = False 
        else:
            for i in range(len(self.equal_weights)):   
                if (round(self.equal_weights[i],2) != round(self.comparison_weights_one[i], 2)):
                    comparison_check_one = False
                    break  
        if (len(self.equal_weights) != len(self.comparison_weights_two)):
            comparison_check_two = False 
        else:
            for i in range(len(self.equal_weights)):   
                if (round(self.equal_weights[i],2) != round(self.comparison_weights_two[i], 2)):
                    comparison_check_two = False
                    break  
        if (len(self.equal_weights) != len(self.comparison_weights_three)):
            comparison_check_three = False 
        else:
            for i in range(len(self.equal_weights)):   
                if (round(self.equal_weights[i],2) != round(self.comparison_weights_three[i], 2)):
                    comparison_check_three = False
                    break  
        if (len(self.equal_weights) != len(self.comparison_weights_four)):
            comparison_check_four = False 
        else:
            for i in range(len(self.equal_weights)):   
                if (round(self.equal_weights[i],2) != round(self.comparison_weights_four[i], 2)):
                    comparison_check_four = False
                    break  
        if (len(self.equal_weights) != len(self.comparison_weights_five)):
            comparison_check_four = False 
        else:
            for i in range(len(self.equal_weights)):   
                if (round(self.equal_weights[i],2) != round(self.comparison_weights_five[i], 2)):
                    comparison_check_five = False
                    break  
                    
        plt.figure(figsize=(10, 6))
        if(self.bm_returns is not None):
            bm_annual_return = self.annualized_return(self.bm_returns, [1])
            bm_annual_volatility = self.annualized_volatility(self.bm_returns, [1])
            plt.plot(bm_annual_volatility, bm_annual_return, color='deeppink', marker='*', markersize=15.0,
                     label='Benchmark')
            
        if(comparison_check_one == False):
            annual_return_one = self.annualized_return(self.our_returns, self.comparison_weights_one)
            annual_volatility_one = self.annualized_volatility(self.our_returns, self.comparison_weights_one)
            plt.plot(annual_volatility_one, annual_return_one, color='mediumpurple', marker='*', markersize=15.0,
                     label=self.portfolio_one_name)
        if(comparison_check_two == False):
            annual_return_two = self.annualized_return(self.our_returns, self.comparison_weights_two)
            annual_volatility_two = self.annualized_volatility(self.our_returns, self.comparison_weights_two)
            plt.plot(annual_volatility_two, annual_return_two, color='indigo', marker='*', markersize=15.0,
                     label=self.portfolio_two_name)
        if(comparison_check_three == False):
            annual_return_three = self.annualized_return(self.our_returns, self.comparison_weights_three)
            annual_volatility_three = self.annualized_volatility(self.our_returns, self.comparison_weights_three)
            plt.plot(annual_volatility_three, annual_return_three, color='plum', marker='*', markersize=15.0,
                     label=self.portfolio_three_name)
        if(comparison_check_four == False):
            annual_return_four = self.annualized_return(self.our_returns, self.comparison_weights_four)
            annual_volatility_four = self.annualized_volatility(self.our_returns, self.comparison_weights_four)
            plt.plot(annual_volatility_four, annual_return_four, color='darkmagenta', marker='*', markersize=15.0,
                     label=self.portfolio_four_name)
        if(comparison_check_four == False):
            annual_return_five = self.annualized_return(self.our_returns, self.comparison_weights_five)
            annual_volatility_five = self.annualized_volatility(self.our_returns, self.comparison_weights_five)
            plt.plot(annual_volatility_five, annual_return_five, color='fuchsia', marker='*', markersize=15.0,
                     label=self.portfolio_five_name)

        if show_true_max:
            maximum_return_weights = self.maximum_return_portfolio()
            maximum_return = self.annualized_return(self.our_returns, maximum_return_weights)
            maximum_return_volatility = self.annualized_volatility(self.our_returns, maximum_return_weights)
            plt.plot(maximum_return_volatility, maximum_return, color='dimgrey', marker='*', markersize=15.0, 
                     label='Maximum Return (within boundaries) Portfolio')
        else:
            ranked_returns = np.array([minimum_risk_return, 
                                       maximum_sharpe_return, 
                                       equal_weight_return, 
                                       bm_annual_return, 
                                       comparison_return_one,
                                       comparison_return_two,
                                       comparison_return_three,
                                       comparison_return_four,
                                       comparison_return_five])
            maximum_return = ranked_returns.max()
        if (maximum_return != minimum_risk_return):
            # It is possible there is no difference between the minimum risk portfolio and maximum return possible
            target_returns = np.linspace(minimum_risk_return, maximum_return, 30)
            target_volatilities = []
            for tret in target_returns:
                res = minimize(lambda weights: MVPortfolio.annualized_volatility(self.our_returns, weights),
                               self.equal_weights,
                               method='SLSQP',
                               bounds=self.bnds,
                               constraints=extra_cons)
                target_volatilities.append(res['fun'])
            target_volatilities = np.array(target_volatilities)
            plt.plot(target_volatilities, target_returns, color='dodgerblue', lw=4.0, label='Efficient Frontier')
        plt.plot(maximum_sharpe_volatility, maximum_sharpe_return,
                 color='gold', marker='*', markersize=20.0, label='Maximum Sharpe Portfolio')
        plt.plot(minimum_risk_volatility, minimum_risk_return,
                  color='forestgreen', marker='*', markersize=10.0, label='Minimum Risk Portfolio')
        plt.plot(equal_weight_volatility, equal_weight_return,
                  color='cyan', marker='*', markersize=10.0, label='Equal Weight Portfolio')
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.legend(loc='upper left')
        plt.title('Efficient Portfolios')
        plt.show();

    def visualize_portfolio_returns(self, start_date=None, end_date=None):
        # Need to format datetimes for fstring in title
        format_string = "%Y-%m-%d"
        if(start_date is None):
            start_date = self.our_returns.index[0].strftime(format_string)
        if(end_date is None):
            end_date = self.our_returns.index[-1].strftime(format_string)
            
        minimum_risk_weights = self.minimum_risk_portfolio()
        minimum_risk_returns = self.cummulative_portfolio_returns(weights=minimum_risk_weights,
                                                                  start_date=start_date,
                                                                  end_date=end_date)
        maximum_sharpe_weights = self.maximum_sharpe_portfolio()
        maximum_sharpe_returns = self.cummulative_portfolio_returns(weights=maximum_sharpe_weights,
                                                                  start_date=start_date,
                                                                  end_date=end_date)
        equal_weight_returns = self.cummulative_portfolio_returns(weights=self.equal_weights,
                                                                  start_date=start_date,
                                                                  end_date=end_date)
        maximum_return_weights = self.maximum_return_portfolio()
        maximum_bounded_returns = self.cummulative_portfolio_returns(weights=maximum_return_weights,
                                                                     start_date=start_date,
                                                                     end_date=end_date)
        
        comparison_check_one = True
        comparison_check_two = True
        comparison_check_three = True
        comparison_check_four = True
        comparison_check_five = True
        if (len(self.equal_weights) != len(self.comparison_weights_one)):
            comparison_check_one = False 
        else:
            for i in range(len(self.equal_weights)):   
                if (round(self.equal_weights[i],2) != round(self.comparison_weights_one[i], 2)):
                    comparison_check_one = False
                    break  
        if (len(self.equal_weights) != len(self.comparison_weights_two)):
            comparison_check_two = False 
        else:
            for i in range(len(self.equal_weights)):   
                if (round(self.equal_weights[i],2) != round(self.comparison_weights_two[i], 2)):
                    comparison_check_two = False
                    break  
        if (len(self.equal_weights) != len(self.comparison_weights_three)):
            comparison_check_three = False 
        else:
            for i in range(len(self.equal_weights)):   
                if (round(self.equal_weights[i],2) != round(self.comparison_weights_three[i], 2)):
                    comparison_check_three = False
                    break  
        if (len(self.equal_weights) != len(self.comparison_weights_four)):
            comparison_check_four = False 
        else:
            for i in range(len(self.equal_weights)):   
                if (round(self.equal_weights[i],2) != round(self.comparison_weights_four[i], 2)):
                    comparison_check_four = False
                    break  
        if (len(self.equal_weights) != len(self.comparison_weights_five)):
            comparison_check_four = False 
        else:
            for i in range(len(self.equal_weights)):   
                if (round(self.equal_weights[i],2) != round(self.comparison_weights_five[i], 2)):
                    comparison_check_five = False
                    break  

        plt.figure(figsize=(10, 6))
        if(self.bm_returns is not None):
            benchmark_returns = self.bm_returns[start_date:end_date].cumsum().apply(np.exp)
            plt.plot(benchmark_returns, 'deeppink', lw=1.0, label='Benchmark')
        if(comparison_check_one == False):
            portfolio_one_returns = self.cummulative_portfolio_returns(weights=self.comparison_weights_one,
                                                                       start_date=start_date,
                                                                       end_date=end_date)
            plt.plot(portfolio_one_returns, 'mediumpurple', lw=1.0, label=self.portfolio_one_name)
        if(comparison_check_two == False):
            portfolio_two_returns = self.cummulative_portfolio_returns(weights=self.comparison_weights_two,
                                                                       start_date=start_date,
                                                                       end_date=end_date)
            plt.plot(portfolio_two_returns, 'indigo', lw=1.0, label=self.portfolio_two_name)
        if(comparison_check_three == False):
            portfolio_three_returns = self.cummulative_portfolio_returns(weights=self.comparison_weights_three,
                                                                       start_date=start_date,
                                                                       end_date=end_date)
            plt.plot(portfolio_three_returns, 'plum', lw=1.0, label=self.portfolio_three_name)
        if(comparison_check_four == False):
            portfolio_four_returns = self.cummulative_portfolio_returns(weights=self.comparison_weights_four,
                                                                       start_date=start_date,
                                                                       end_date=end_date)
            plt.plot(portfolio_four_returns, 'darkmagenta', lw=1.0, label=self.portfolio_four_name)
        if(comparison_check_five == False):
            portfolio_five_returns = self.cummulative_portfolio_returns(weights=self.comparison_weights_five,
                                                                       start_date=start_date,
                                                                       end_date=end_date)
            plt.plot(portfolio_five_returns, 'fuchsia', lw=1.0, label=self.portfolio_five_name)
        plt.plot(maximum_sharpe_returns, 'gold', lw=1.0, label='Maximum Sharpe Portfolio')
        plt.plot(minimum_risk_returns, 'forestgreen', lw=1.0, label='Minimum Risk Portfolio')
        plt.plot(equal_weight_returns, 'cyan', lw=1.0, label='Equal Weight Portfolio')
        plt.plot(maximum_bounded_returns, 'dimgrey', lw=1.0, label='Maximum Return (within boundaries) Portfolio')
        plt.xlabel('Time')
        plt.ylabel('Cummulative Return')
        plt.legend(loc='upper left')
        plt.title(f"Efficient Portfolio Returns from {start_date} to {end_date}")
        plt.show();

    
        