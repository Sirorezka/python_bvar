import time
from typing import Dict, Literal
from copy import deepcopy
import pandas as pd
import numpy as np
import pymc as pm

def create_bvar_model(data: pd.DataFrame, 
                      n_eq: int, 
                      n_train: int, 
                      n_lags: int, 
                      priors: Dict, 
                      sampler_params: Dict,  
                      mv_norm: bool = True):
    """
        Arguments:
            n_lags - Number of lags to use
            n_eq - Number of equations to use in VAR. All other variables will have lags, but won't be modelled
            n_train - number of observations to user for training, you can use negative value to train on full dataset
            mv_norm - use multivariate normal distribution to estimate covariance matrix between equations 
            sampler_params - parameters to pass to PYMC sampler
            priors - list of priors

    """

    
    assert n_eq>=1 
    
    if n_train>0:
        df = data.iloc[:n_train].copy(deep=True)
    else:
        df = data.copy(deep=True)
            
    n_obs = df.shape[0]    
    n_vars = df.shape[1]    


    coords = {
        "lags": np.arange(n_lags) + 1,
        "equations": df.columns[:n_eq].tolist(),
        "cross_vars": [f"var_{x}" for x in df.columns.tolist()],
        "time": [i for i in df.index[n_lags:]],
    }


    with pm.Model(coords=coords) as model:

        # N_eq equations, each with N_lags and N_eq variables (to which lags will be applied)        
        lag_coefs = pm.Normal(
            "lag_coefs",
            mu=priors["lag_coefs"]["mu"],
            sigma=priors["lag_coefs"]["sigma"],
            dims=["equations", "lags", "cross_vars"],
        )
    
        alpha = pm.Normal(
            "alpha", mu=priors["alpha"]["mu"], sigma=priors["alpha"]["sigma"], shape = [1, n_eq]
        )

        data_obs = pm.Data(f"data_obs", 
                           df.iloc[n_lags:,:n_eq].values,  
                           mutable=True)
        
        data_lags = {}
        n_obs = df.shape[0]
        for i in range(n_lags):
            dl = pm.Data(f"data_lag{i+1}", 
                         df.iloc[n_lags-i-1:n_obs-i-1].values,  
                         mutable=True)

            print(dl.shape.eval())
            data_lags[i] = dl
        
        # Create VAR equations
        var = []
        # returns tensor with dimension [n_train_obs - n_lags, 1]
        for j in range(n_eq):

            ar = pm.math.sum(
                [
                pm.math.sum(lag_coefs[j, i] * data_lags[i], axis=-1) 
                    for i in range(n_lags)
                ], axis=0)
            print ('ar_eq:', ar.shape.eval())
            var.append(ar)
    
        # beta multiplied by X
        # shape = [n_train_obs - n_lags, n_eq]
        betaX = pm.math.stack(var, axis=-1)
    
        print ('betax:', betaX.shape.eval())

        mean = pm.Deterministic(
            "mean_equation",
             betaX + alpha,
        )
        print ('mean:', mean.shape.eval())

        
        n_out = data_lags[0].shape[0]
        
        if mv_norm:
            # https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.LKJCholeskyCov.html
            chol, _, _ = pm.LKJCholeskyCov(
                "noise_chol",
                eta=priors["noise_chol"]["eta"],
                n=n_eq,
                sd_dist=pm.HalfNormal.dist(sigma=priors["noise_chol"]["sigma"],shape=n_eq), 
            )

            cov = pm.math.dot(chol, chol.T)
            
            # print ("noise_chol", chol.shape.eval())

            # obs = pm.MvNormal(
            #     "obs", mu=mean, cov=cov, observed=data_obs, shape=data_lags[0].shape)

            
            # В общем тут хак, чтобы сымитировать мультивариантное нормальное распредление 
            # дефолтная имплементация падает.
            vals_raw = pm.Normal("vals_raw", mu=mean, sigma=1, shape=[n_out, n_eq])
            vals = pm.math.dot(vals_raw, chol)

            
            obs = pm.Normal("obs", mu=vals, sigma=0.0001, observed=data_obs, shape=[n_out, n_eq])
            print ('obs', obs.shape.eval())
        
        else:
            ## This is an alternative likelihood that can recover sensible estimates of the coefficients
            ## But lacks the multivariate correlation between the timeseries.
            sigma = pm.HalfNormal("noise", sigma=priors["noise"]["sigma"], dims=["equations"])
                
            print ('sigma:', sigma.shape.eval())
            print ('data_obs:', data_obs.shape.eval())        
            obs = pm.Normal(
                "obs", mu=mean, sigma=sigma, observed=data_obs, shape=[n_out, n_eq])

            # печатаем сколько параметров оценивали
        for rv, shape in model.eval_rv_shapes().items():
            print(f"{rv:>11}: shape={shape}")

        
        st_time = time.time()
        
        idata = None        
        idata = pm.sample(**sampler_params)     
        
        duration = round(time.time() - st_time,1)
        print (f"Total train duration: {duration}s")
        
    return model, idata


def predict_bvar_model(df: pd.DataFrame, 
                  model, 
                  idata = None,
                  inference_type: Literal['prior','posterior'] = 'posterior',
                  plot_graphs: bool = True):
    
    df_test = df.copy(deep=True)
    
    n_obs = df_test.shape[0]
    n_eq, n_lags, n_vars = model.lag_coefs.shape.eval()

    print ("INFERENCE PARAMS")
    print ("inference type:", inference_type)
    print ("Count equations:", n_eq)    
    print ("Count lags:", n_lags)

    with deepcopy(model):
        
        for i in range(n_lags):
            pm.set_data({f"data_lag{i+1}": df_test.iloc[n_lags-i-1:n_obs-i-1].values})

        if inference_type == 'prior':
            trace = pm.sample_prior_predictive(random_seed=44)
            results = trace.prior_predictive.obs.mean(('chain', 'draw')).values
        else:
            assert (idata is not None), ValueError("Argument 'idata' can't be None for 'posterior' inference")
            trace = pm.sample_posterior_predictive(idata, random_seed=44, predictions=True)
            results = trace.predictions['obs'].mean(('chain', 'draw')).values
            # posterior

    df_preds = pd.DataFrame(results, 
                            columns = df.columns[:n_eq],
                            index = df.iloc[n_lags:].index.values)       
    
    if not plot_graphs:
        return df_preds
    
    for i in range(n_eq):
        col_name = df.columns[i]
        df_preds = pd.DataFrame({f'{col_name}_tru': df[col_name].values[n_lags:], # первые два наблюдения съели лаги
                                 f'{col_name}_pred': results[:,i] # первая перемення gdp
                                })
        df_preds.plot()

     
    return df_preds



def predict_bvar_forward_step(df, model, idata):
    """ Take most recent data and predict one step forward. """
    df_test = df.copy(deep=True)    
    n_obs = df_test.shape[0]
    n_eq, n_lags, n_vars = model.lag_coefs.shape.eval()

    print ("INFERENCE PARAMS")
    print ("inference type:", 'one_step_forward_forecast')
    print ("Count equations:", n_eq)    
    print ("Count lags:", n_lags)

    with deepcopy(model):
        
        n_obs = df.shape[0]
        
        for i in range(n_lags):
            # первый лаг должен взять самое последнее наблюдение
            tt = df.iloc[n_obs-i-1:n_obs-i]
            pm.set_data({f"data_lag{i+1}": tt})
            

        id_step = pm.sample_posterior_predictive(idata, random_seed=44, predictions=True)

    prediction = id_step.predictions['obs'].mean(('chain', 'draw')).values

    n_eq = prediction.shape[1]

    df_preds = pd.DataFrame(prediction, columns  = df.columns[:n_eq], index= ['forecast'])    
    return df_preds