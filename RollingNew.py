import numpy as np
import pandas as pd
import pyfolio as pf

class SLMAMomentum():
    def __init__(self, prices, SMA, LMA, holding_period, cost=0.001, weightings='emv', long_only=False, show_analytics=True):   
        self.returns = self.get_returns(prices)
        self.holding_returns = self.get_holding_returns(prices, holding_period)
        
        print(self.returns)
        
        self.signal = self.rolling_momentum(prices, SMA, LMA, long_only)

        if weightings == 'ew':
            self.cs_risk_weight = self.equal_weight(self.signal)
        elif weightings == 'emv':
            self.cs_risk_weight = self.equal_marginal_volatility(self.returns, self.signal)

        self.rebalance_weight = 1 / holding_period
        self.cost = self.transaction_cost(self.signal, cost)

        self.port_rets_wo_cash = self.backtest(self.holding_returns, self.signal, self.cost, self.rebalance_weight, self.cs_risk_weight)
        
        self.ts_risk_weight = self.volatility_targeting(self.port_rets_wo_cash)
        
        self.port_rets = self.port_rets_wo_cash * self.ts_risk_weight
        
        if show_analytics == True:
            self.performance_analytics(self.port_rets)                          
                
    def get_returns(self, prices):
        """Returns the historical daily returns
        
        Paramters
        ---------
        prices : dataframe
            Historical daily prices
            
        Returns
        -------
        returns : dataframe
            Historical daily returns
        """
        returns = prices.pct_change().fillna(0)
        return returns

    def get_holding_returns(self, prices, holding_period):
        """Returns the periodic returns for each holding period
        
        Paramters
        ---------
        returns : dataframe
            Historical daily returns
        holding_period : int
            Holding Period
            
        Returns
        -------
        holding_returns : dataframe
            Periodic returns for each holding period. Pulled by N (holding_period) days forward to keep inline with trading signals.
        """
        holding_returns = prices.pct_change(periods=holding_period).shift(-holding_period).fillna(0)
        return holding_returns

    def rolling_momentum(self, prices, SMA, LMA, long_only=False):
        """Returns Absolute Momentum Signals
        
        Parameters
        ----------
        prices : dataframe
            Historical daily prices
        SMA : int
            Short-term Moving Average

        LMA : int
             Long-term Moving Average
        long_only : bool, optional
            Indicator for long-only momentum, False is default value
        
        Returns
        -------
        returns : dataframe
            Absolute momentum signals     
        """    
        rollingSMA = prices.rolling(SMA).mean()
        rollingLMA = prices.rolling(LMA).mean()
        long_signal = (rollingSMA > rollingLMA).applymap(self.bool_converter)
        short_signal = -(rollingSMA <rollingLMA).applymap(self.bool_converter)    
        if long_only == True:
            signal = long_signal
        else:
            signal = long_signal + short_signal
        return signal
    


    def equal_weight(self, signal):
        """Returns Equal Weights
        Parameters
        ----------
        signal : dataframe
            Momentum signal dataframe
        Returns
        -------
        weight : dataframe
            Equal weights for cross-asset momentum portfolio
        """
        total_signal = 1 / abs(signal).sum(axis=1)
        total_signal.replace([np.inf, -np.inf], 0, inplace=True)
        weight = pd.DataFrame(index=signal.index, columns=signal.columns).fillna(value=1)
        weight = weight.mul(total_signal, axis=0)
        return weight

    def equal_marginal_volatility(self, returns, signal):
        """Returns Equal Marginal Volatility (Inverse Volatility)
        
        Parameters
        ----------
        returns : dataframe
            Historical daily returns
        signal : dataframe
            Momentum signal dataframe
        Returns
        -------
        weight : dataframe
            Weights using equal marginal volatility
        """
        vol = (returns.rolling(252).std() * np.sqrt(252)).fillna(0)
        vol_signal = vol * abs(signal)
        inv_vol = 1 / vol_signal
        inv_vol.replace([np.inf, -np.inf], 0, inplace=True)
        weight = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)
        return weight

    def volatility_targeting(self, returns, target_vol=0.01):
        """Returns Weights based on Vol Target
        
        Parameters
        ----------
        returns : dataframe
            Historical daily returns of backtested portfolio
        target_vol : float, optional
            Target volatility, Default target volatility is 1%
        Returns
        -------
        weights : dataframe
            Weights using equal marginal volatility
        """
        weight = target_vol / (returns.rolling(252).std() * np.sqrt(252)).fillna(0)
        weight.replace([np.inf, -np.inf], 0, inplace=True)
        weight = weight.shift(1).fillna(0)
        return weight

    def transaction_cost(self, signal, cost=0.001):
        """Returns Transaction Costs
        
        Parameters
        ----------
        signal : dataframe
            Momentum signal dataframe
        cost : float, optional
            Transaction cost (%) per each trade. The default is 0.001.
        Returns
        -------
        cost_df : dataframe
            Transaction cost dataframe
        """
        cost_df = (signal.diff() != 0).applymap(self.bool_converter) * cost
        cost_df.iloc[0] = 0
        return cost_df
    
    def backtest(self, returns, signal, cost, rebalance_weight, weighting):
        """Returns Portfolio Returns without Time-Series Risk Weights
        Parameters
        ----------
        returns : dataframe
            Historical daily returns
        signal : dataframe
            Momentum signal dataframe
        cost : dataframe
            Transaction cost dataframe
        rebalance_weight : float
            Rebalance weight
        weighting : dataframe
            Weighting dataframe
        Returns
        -------
        port_rets : dataframe
            Portfolio returns dataframe without applying time-series risk model
        """
        port_rets = ((signal * returns - cost) * rebalance_weight * weighting).sum(axis=1)
        return port_rets

    def performance_analytics(self, returns):
        """Returns Perforamnce Analytics using pyfolio package
        Parameters
        ----------
        returns : series
            backtestd portfolio returns
        Returns
        -------
        None
        """
        pf.create_returns_tear_sheet(returns)

    def bool_converter(self, bool_var):
        """Returns Integer Value from Boolean Value
        Parameters
        ----------
        bool_var : boolean
            Boolean variables representing trade signals
        Returns
        -------
        result : int
            Integer variables representing trade signals
        """
        if bool_var == True:
            result = 1
        elif bool_var == False:
            result = 0
        return result

def get_price_df(url,choice = 'every'):
    """Returns price dataframe from given URL
    Parameters
    ----------
    url : string
        URL which contains dataset

    choice : string
        Defaulted as 'every' for every asset, type a string for a specific product

    Returns
    -------
    df : dataframe
        Imported price dataframe from URL
    """
    df = pd.read_csv(url).dropna()
    df.index = pd.to_datetime(df['Date'])
    df = df.drop(columns=['Date'])
    if choice != 'every':
        df = df[[choice]]
    return df
