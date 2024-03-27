from scipy import linalg
import pandas as pd
import numpy as np

def estimate_ols_coefs(df: pd.DataFrame, 
                       n_train:int, 
                       n_eq: int, 
                       n_lags: int):
    """
        Returns
        - alfa: 'free term' in the regression, size of [n_eq]
        - lag_coefs: array size of [n_eq, n_lags, n_vars]
    """

    df = df.iloc[:n_train].copy(deep=True)
    n_obs = df.shape[0]
    n_vars = df.shape[1]

    alfa = [] 
    lag_coefs = []
    
    for i in range(n_eq):    
        a = np.ones([n_obs-n_lags,1])
        for j in range(n_lags):
            var_lag = df.iloc[n_lags-j-1:n_obs-j-1].values        
            a = np.concatenate([a,var_lag],axis=-1)
        
            
        b = df.iloc[n_lags:,i].copy()
        sol = linalg.lstsq(a,b)
        coef = sol[0]
    
        alfa.append(coef[0])
    
        # reshapping coefficients
        # out shape -  [1, n_lags, n_vars]
        lag_coef = coef[1:].reshape([1,n_lags,-1])
        lag_coefs.append(lag_coef)
    
    alfa = np.array(alfa)    
    lag_coefs = np.concatenate(lag_coefs, axis=0)

    return alfa, lag_coefs

