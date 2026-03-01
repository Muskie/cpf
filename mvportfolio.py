import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from scipy.optimize import minimize
from prettytable import PrettyTable
from datetime import datetime
# I tried including these in the Jupyter Notebook as it may help with rendering in Google Colab
plt.style.use('seaborn-v0_8')
pd.set_option("display.precision", 5)
np.set_printoptions(suppress=True,
        formatter={'float': lambda x: f'{x:.4f}'})

import warnings
warnings.filterwarnings('ignore') # I get a RuntimeWarning: Mean of empty slice. Or did so I added this.


class MVPortfolio:
    """Mean Variance Portfolio Class"""
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
        """Mean Variance Portfolio Initiation Method"""
        # Some more checking of the passed in returns would be wise.
        self.universe=universe
        self.primary_exchange=primary_exchange
        if holdings is None:
            self.holdings = self.universe.columns[:]
        else:
            self.holdings = holdings
        self.our_returns=self.universe[self.holdings]
        # To do some calculations properly you need a calendar
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
        # I could have made a dictionary perhaps instead of doing it like this
        # This prevents calling methods on empty arrays accidently
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
        """Set the calendar and key dates used in MVPortfolio class."""
        self.calendar = mcal.get_calendar(self.primary_exchange)
        
        first_date_of_data = self.our_returns.index[0]
        first_year_of_data = first_date_of_data.year
        year_start_date = datetime(first_year_of_data, 1, 1)
        year_end_date = datetime(first_year_of_data, 12, 31)
        schedule = self.calendar.schedule(start_date=year_start_date, end_date=year_end_date)
        if not schedule.empty:
            self.first_yearly_trading_date = schedule.index[0].to_pydatetime()
        # This will advance us one year
        if(self.first_yearly_trading_date < first_date_of_data):
            next_year_of_data = first_year_of_data + 1
            year_start_date = datetime(next_year_of_data, 1, 1)
            year_end_date = datetime(next_year_of_data, 12, 31)
            schedule = self.calendar.schedule(start_date=year_start_date, end_date=year_end_date)
            self.first_yearly_trading_date = schedule.index[0].to_pydatetime()

        last_date_of_data = self.our_returns.index[-1]
        last_year_of_data = last_date_of_data.year
        year_start_date = datetime(last_year_of_data, 1, 1)
        year_end_date = datetime(last_year_of_data, 12, 31)
        schedule = self.calendar.schedule(start_date=year_start_date, end_date=year_end_date)
        if not schedule.empty:
            self.last_yearly_trading_date = schedule.index[-1].to_pydatetime()
        # This will reduce one year but perhaps isn't perfect, but it is safe
        if(self.last_yearly_trading_date.month > last_date_of_data.month):
            prev_year_of_data = last_year_of_data - 1
            year_start_date = datetime(prev_year_of_data, 1, 1)
            year_end_date = datetime(prev_year_of_data, 12, 31)
            schedule = self.calendar.schedule(start_date=year_start_date, end_date=year_end_date)
            self.last_yearly_trading_date = schedule.index[-1].to_pydatetime()
        
    def _determine_asset_classes(self,
                                 asset_class_weights=None):
        """Determine which of equity, debt, and alternative asset classes are in use."""
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
                self.asset_class_weights = asset_class_weights
        else:
            self.asset_class_weights = asset_class_weights

    # Target asset class weights can be overriden by passing in exacting boundaries (bnds)
    def target_equity_weight(self):
        """Accessor for target equity weight."""
        return self.asset_class_weights[0]

    def target_debt_weight(self):
        """Accessor for target debt weight."""
        return self.asset_class_weights[1]

    def target_alt_weight(self):
        """Accessor for target alternative asset weight."""
        return self.asset_class_weights[2]

    def _set_boundaries_and_constraints(self,
                                        min_largest_asset_class_position_size=0.0,
                                        max_largest_asset_class_position_size=1.0,
                                        boundaries=None,
                                        constraints=None):
        """Sets the boundaries and constraints applied to the portfolio and algorithms."""
        # I let people override contraints and boundaries, constraints is simple, boundaries is convoluted.
        if constraints is None:
            self.cons = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}) 
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
        """Annualized return of the weights and returns."""
        return np.dot(rets.mean(), weights) * 252  

    @staticmethod
    def annualized_volatility(rets, weights):
        """Annualized volatility of the weights and returns."""
        return math.sqrt(np.dot(weights, np.dot(rets.cov() * 252 , weights)))

    @classmethod
    def sharpe_ratio(cls, rets, weights):
        """Annualized Sharpe Ratio of the weights and returns."""
        return cls.annualized_return(rets, weights) / cls.annualized_volatility(rets, weights)

    def maximum_return_portfolio(self):
        """The weights which lead to the maximum return within boundaries and constraints."""
        opt = minimize(lambda weights: -MVPortfolio.annualized_return(self.our_returns, weights),
                       self.equal_weights,
                       bounds=self.bnds,
                       constraints=self.cons)
        return opt['x']

    def minimum_risk_portfolio(self):
        """The minimum risk portfolio that is still efficient."""
        opt = minimize(lambda weights: MVPortfolio.annualized_volatility(self.our_returns, weights),
                       self.equal_weights, 
                       bounds=self.bnds, 
                       constraints=self.cons)
        return opt['x']

    def maximum_sharpe_portfolio(self):
        """The weights that maximize the Sharpe Ratio."""
        opt = minimize(lambda weights: -MVPortfolio.sharpe_ratio(self.our_returns, weights),
                       self.equal_weights,
                       bounds=self.bnds,
                       constraints=self.cons)
        return opt['x']

    def dataframe_for_weights(self, weights):
        """Displayes a pandas dataframe of asset weights."""
        data = {'Asset': self.holdings,
                'Weight': weights}
        df = pd.DataFrame(data)
        df_indexed = df.set_index('Asset')
        return df_indexed


    def daily_portfolio_returns(self, weights, start_date=None, end_date=None):
        """The daily portfolio returns for the weights between two dates."""
        if(start_date is None):
            start_date = self.our_returns.index[0]
        if(end_date is None):
            end_date = self.our_returns.index[-1]
        return self.our_returns[start_date:end_date].mul(weights, axis=1).sum(axis=1)

    def cummulative_portfolio_returns(self, weights, start_date=None, end_date=None):
        """The cummulative portfolio returns for given weights between two dates."""
        if(start_date is None):
            start_date = self.our_returns.index[0]
        if(end_date is None):
            end_date = self.our_returns.index[-1]
        daily_portfolio_returns = self.daily_portfolio_returns(weights, start_date, end_date)
        return daily_portfolio_returns.cumsum().apply(np.exp)

    def calendar_year_maximum_sharpe_portfolios(self):
        """The return of the maximum Sharpe Ratio portfolio for each complete calendar year."""
        # I tried using pandas_market_calendars in the constructor Google Colab added another hoop but it works
        year_one = self.first_yearly_trading_date.year
        last_year = self.last_yearly_trading_date.year
        last_year_plus_one = last_year + 1
        optimal_yearly_weights = {}
        for year in range(year_one, last_year_plus_one):
            rets_ = self.our_returns.loc[f'{year}-01-01':f'{year}-12-31']
            ow = minimize(lambda weights: -MVPortfolio.sharpe_ratio(rets_, weights),
                               self.equal_weights,
                               bounds=self.bnds,
                               constraints=self.cons)['x']
            optimal_yearly_weights[year] = ow
        return optimal_yearly_weights

    def yearly_returns(self, weights):
        """The return for the weights for each complete calendar year."""
        # This will take iterable of weights and build the dictionary I can pass to the next method
        year_one = self.first_yearly_trading_date.year
        last_year = self.last_yearly_trading_date.year
        last_year_plus_one = last_year + 1
        yearly_returns = {}
        for year in range(year_one, last_year_plus_one):
            rets_ = self.our_returns.loc[f'{year}-01-01':f'{year}-12-31']
            yearly_returns[year] = self.annualized_return(rets_, weights)
        return yearly_returns

    def pretty_yearly_weights(self, yearly_weights):
        """Displays a PrettyTable of the yearly weights."""
        last_year = self.last_yearly_trading_date.year
        yearly_weight_table = PrettyTable(["Year", "Holding", "Weights"])
        for key, val in yearly_weights.items():
            for holding in range(self.noa):
                yearly_weight_table.add_row([key, self.holdings[holding], val[holding].round(2)]) # Round for display
            empty_row = [" "] * len(yearly_weight_table.field_names)
            if(key != last_year): # This prevents the last empty row
                yearly_weight_table.add_row(empty_row) 
        print(yearly_weight_table)

    def visualize_efficient_frontier(self, show_true_max=True):
        """Returns a Matplotlib plot of the efficient frontier and various portfolios."""
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
        # For demoability I do this five times, the only advantage is preventing stars on top of stars
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
                    
        # Google Colab requires adjusting my formatting
        plt.figure(figsize=(6, 3)) 
        plt.rc('legend', fontsize='xx-small')
        if(self.bm_returns is not None):
            bm_annual_return = self.annualized_return(self.bm_returns, [1])
            bm_annual_volatility = self.annualized_volatility(self.bm_returns, [1])
            plt.plot(bm_annual_volatility, bm_annual_return, color='deeppink', marker='*', markersize=12.0,
                     label='Benchmark')
            
        if(comparison_check_one == False):
            annual_return_one = self.annualized_return(self.our_returns, self.comparison_weights_one)
            annual_volatility_one = self.annualized_volatility(self.our_returns, self.comparison_weights_one)
            plt.plot(annual_volatility_one, annual_return_one, color='mediumpurple', marker='*', markersize=12.0,
                     label=self.portfolio_one_name)
        if(comparison_check_two == False):
            annual_return_two = self.annualized_return(self.our_returns, self.comparison_weights_two)
            annual_volatility_two = self.annualized_volatility(self.our_returns, self.comparison_weights_two)
            plt.plot(annual_volatility_two, annual_return_two, color='indigo', marker='*', markersize=12.0,
                     label=self.portfolio_two_name)
        if(comparison_check_three == False):
            annual_return_three = self.annualized_return(self.our_returns, self.comparison_weights_three)
            annual_volatility_three = self.annualized_volatility(self.our_returns, self.comparison_weights_three)
            plt.plot(annual_volatility_three, annual_return_three, color='plum', marker='*', markersize=12.0,
                     label=self.portfolio_three_name)
        if(comparison_check_four == False):
            annual_return_four = self.annualized_return(self.our_returns, self.comparison_weights_four)
            annual_volatility_four = self.annualized_volatility(self.our_returns, self.comparison_weights_four)
            plt.plot(annual_volatility_four, annual_return_four, color='darkmagenta', marker='*', markersize=12.0,
                     label=self.portfolio_four_name)
        if(comparison_check_four == False):
            annual_return_five = self.annualized_return(self.our_returns, self.comparison_weights_five)
            annual_volatility_five = self.annualized_volatility(self.our_returns, self.comparison_weights_five)
            plt.plot(annual_volatility_five, annual_return_five, color='fuchsia', marker='*', markersize=12.0,
                     label=self.portfolio_five_name)

        if show_true_max:
            maximum_return_weights = self.maximum_return_portfolio()
            maximum_return = self.annualized_return(self.our_returns, maximum_return_weights)
            maximum_return_volatility = self.annualized_volatility(self.our_returns, maximum_return_weights)
            plt.plot(maximum_return_volatility, maximum_return, color='dimgrey', marker='*', markersize=12.0, 
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
                 color='gold', marker='*', markersize=14.0, label='Maximum Sharpe Portfolio')
        plt.plot(minimum_risk_volatility, minimum_risk_return,
                  color='forestgreen', marker='*', markersize=10.0, label='Minimum Risk Portfolio')
        plt.plot(equal_weight_volatility, equal_weight_return,
                  color='cyan', marker='*', markersize=7.0, label='Equal Weight Portfolio')
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.legend(loc='upper left')
        plt.title('Efficient Portfolios')
        plt.show();

    def visualize_portfolio_returns(self, start_date=None, end_date=None):
        """Returns the cummulative returns as a Matplotlib line graph."""
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

        # Google Colab requires adjusting my formatting
        plt.figure(figsize=(6, 3)) 
        plt.rc('legend', fontsize='xx-small')
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
        plt.plot(maximum_bounded_returns, 'dimgrey', lw=1.0, label='Maximum Return Efficient Portfolio')
        plt.xlabel('Time', fontsize='x-small')
        plt.ylabel('Cummulative Return')
        plt.legend(loc='upper left')
        plt.title(f"Efficient Portfolio Returns from {start_date} to {end_date}")
        plt.show();

    def summary_dataframe_for_weights(self, portfolio_weights, portfolio_name='Portfolio'):
        """Returns a Pandas dataframe detailing various aspects of the portfolio using the passed in weights."""
        portfolio_values = []
        portfolio_return = self.annualized_return(self.our_returns, portfolio_weights)
        portfolio_volatility = self.annualized_volatility(self.our_returns, portfolio_weights)
        portfolio_sharpe_ratio = self.sharpe_ratio(self.our_returns, portfolio_weights)
        portfolio_yearly_returns = self.yearly_returns(portfolio_weights)
        portfolio_values.append(portfolio_return)
        portfolio_values.append(portfolio_volatility)
        portfolio_values.append(portfolio_sharpe_ratio)
        portfolio_values.extend(portfolio_weights)
        portfolio_values.extend(list(portfolio_yearly_returns.values()))

        my_index = ["Annualized Return", "Annualized Volatility", "Annualized Sharpe Ratio"]
        my_index.extend(self.holdings)
        years = list(portfolio_yearly_returns.keys())
        my_index.extend(years)

        d = {portfolio_name: pd.Series(portfolio_values, index=my_index)}
        df = pd.DataFrame(d)

        return df

    def summary(self):
        """Returns a dataframe summarizing various aspects of various portfolios."""
        # Four Portfolios are always calculated, five more are potentially calculated
        summary_df = pd.DataFrame()

        maximum_sharpe_weights = self.maximum_sharpe_portfolio()
        summary_df = pd.concat((summary_df,
                               self.summary_dataframe_for_weights(maximum_sharpe_weights,
                                                                  'Maximum Sharpe Portfolio')),
                               axis=1)
                               
        minimum_risk_weights = self.minimum_risk_portfolio()
        summary_df = pd.concat((summary_df, 
                               self.summary_dataframe_for_weights(minimum_risk_weights,
                                                                  'Minimum Risk Portfolio')),
                               axis=1)

        summary_df = pd.concat((summary_df,
                               self.summary_dataframe_for_weights(self.equal_weights,
                                                                  'Equal Weight Portfolio')),
                               axis=1)
        
        maximum_return_weights = self.maximum_return_portfolio()
        summary_df = pd.concat((summary_df,
                               self.summary_dataframe_for_weights(maximum_return_weights,
                                                                  'Maximum Return Efficient Portfolio')),
                               axis=1)
        
        # For demoability I do this five times and keep sticking with this design quirk       
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
            
        if(comparison_check_one == False):
            summary_df = pd.concat((summary_df,
                                   self.summary_dataframe_for_weights(self.comparison_weights_one,
                                                                      self.portfolio_one_name)),
                                   axis=1)
        if(comparison_check_two == False):
            summary_df = pd.concat((summary_df,
                                   self.summary_dataframe_for_weights(self.comparison_weights_two,
                                                                      self.portfolio_two_name)),
                                   axis=1)
        if(comparison_check_three == False):
            summary_df = pd.concat((summary_df,
                                   self.summary_dataframe_for_weights(self.comparison_weights_three,
                                                             self.portfolio_three_name)),
                                   axis=1)
        if(comparison_check_four == False):
            summary_df = pd.concat((summary_df,
                                   self.summary_dataframe_for_weights(self.comparison_weights_four,
                                                                      self.portfolio_four_name)),
                                   axis=1)
        if(comparison_check_four == False):
            summary_df = pd.concat((summary_df,
                                   self.summary_dataframe_for_weights(self.comparison_weights_five,
                                                                      self.portfolio_five_name)),
                                   axis=1)

        return summary_df
    
        