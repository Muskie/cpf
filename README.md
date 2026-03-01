# Balanced Portfolio Asset Allocation with Sckit-Learn and PyTorch

Code and data for my *Certificate in Python for Finance* final project:

By [Andrew Muschamp McKay](https://www.muschamp.ca/)

All experimental results can be reproduced using the code and data in this repository.

**Abstract**

The 60% equity and 40% debt portfolio is a classic of the asset management industry and retirement planning for generations. Although there a proven benefits to diversification and choosing assets which have an uncorrelated time series of daily price or return movements. Simply using a heuristic or rule-of-thumb to allocate client assets is not always optimal. So continuing on from the Python for Asset Management class and the Reinforcement Learning for Finance class and book in the [Python for Finance](https://python-for-finance.com) certificate program, Scikit_Learn and PyTorch were employed to see if machine learning could allocate assets more effectively.

Clients often have an an Investment Policy Statement (IPS) which details which assets they are willing to hold and in what proportion. This can be modeled in Python and an Investing Agent can be made to adhere to these pre-determined boundaries. Success will be determined primarily by the Sharpe Ratio which combines risk and return into a single number **where a larger number is clearly better**. Can the Investing Agent beat the average 60:40 portfolio, can it beat the benchmark detailed in the IPS? *Let's find out.*

The assets chosen are publicly available in Canada and the prices and returns used are calculated by State Street. 

**Requirments** 

* Python version 3.12.2
* numpy, pandas, torch, scipy, matplotlib, pandas_market_calendar, prettytable, seaborn

**Acknowledgements**

Thanks to Yves Hilpisch for his books, classes and mentorship. Thanks also to the greater online Python community and open source project contributors.
