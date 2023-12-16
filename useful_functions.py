import xarray as xr 
import tensorflow as tf 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.stats import wilcoxon

# computes quantile ranges across each pixel first
def eval_model_stat(X, y, models, stat_map, Q_range=None, aux_data=None):
    # each item will be the cumulative statistic for each model
    stat_list = []

    # if no quantile range, evaluate on whole dataset
    if Q_range is None:
        Xq, yq_true = X, y
    else:
        qlo, qhi = Q_range
        Xq, yq_true = data_between(y, qlo, X, qhi)

    # compute vector of averages across time domain
    for model in models:
        spatial_stat = np.array([])  # collect here in average of each pixel across time domain
        yq_pred = model(Xq)
        spatial_stat = eval_pred_stat(yq_true, yq_pred, stat_map, aux_data=aux_data)
        stat_list.append(spatial_stat)

    # returns a list of stats for each model
    return stat_list
    

def data_between(y, q_low, X=None, q_high=None):
    # get the datapoints within a quantile range, measured from cumculative sum on y
    # if q_high is None, then select only the y closest to q_low
    sum_array = np.sort(np.sum(y, axis=-1))
    N = len(sum_array)
    
    # collect indexes for low and high sum boundaries 
    # round of i_low, i_high is the 'nearest' method, effectively qval=0.7666 is 0.76 for example
    # interpolation is not feasiable in this setting
    i_low = min(int(q_low * N), N - 1)
    if q_high is not None:
        i_high = min(int(q_high * N), N - 1)
        
    y_low_sum = sum_array[i_low]
    if q_high is None:
        y_high_sum = y_low_sum
    else:
        y_high_sum = sum_array[i_high] 
    
    y_low_sum = np.round(y_low_sum)
    y_high_sum = np.round(y_high_sum)
    
    mask = [True if y_low_sum <= np.round(np.sum(yi)) <= y_high_sum else False for yi in y]
    if X is not None:
        return X[mask], y[mask]   
    else:
        return y[mask]
    

# put elements of dicts in dict_list1 into dictionaries corresponding by index in dict_list2
def concat_dict(dict_list1, dict_list2):
    for dict1, dict2 in zip(dict_list1, dict_list2):
        for key1, value1 in dict1.items():
            dict2[key1] = value1
    
    return dict_list2
    
    
def MSE(y_true, y_pred):
    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_true, y_pred))


def qassess(y_true, y_pred, q_low, q_high, stat_map):
    # use ground truth to split on this 
    yqt, yqp = data_between(y_true, q_low, y_pred, q_high) 
    
    return stat_map(yqt, yqp) 

# mean of list of xarrays, avoids conversion to numpy array
def xarray_mean(preds, new_names={'dim_0': 'lon', 'dim_1':'lat'}):
    xarr = xr.DataArray(np.sum(preds, axis=0))
    xarr *= 1 / len(preds)
    #xarr = xarr.rename(new_names)

    return xarr


# remove the nan values of the image, which is the ocean around north and south island 
def drop_nan_values(data, dim0='longitude', dim1='latitude'):
    data = data.rename({dim0: 'lon', dim1:'lat'})
    data = data.stack(z=['lat','lon']).dropna('z')  
    
    return data 


# evaluate, unstack then average over runs 
# if model is actually just the obs data, don't collect list of predictions     
def evaluate_models(
                        models, 
                        inputs, 
                        coords=None, 
                        lat=None, 
                        lon=None, 
                        take_mean=True,
                        stack=True
                    ):
    model_out = []
    # if not None, take mean of prediction
    for n, model in enumerate(models):
        pred = model(inputs)
        if coords is not None:
            pred = unstack(pred, coords)
            if lat is not None and lon is not None:
                pred = pred.sel(lat=lat, lon=lon)
            if stack:
                pred = pred.stack(z=['lat','lon']).dropna('z')  
            
        model_out.append(pred)
            
    if take_mean:
        return xarray_mean(model_out) 
    else:
        return model_out
        
    
def unstack(array, coords, do_squeeze=False):
    time_coord = coords['time']
    del coords['time'] 
    array = xr.DataArray(array, coords=coords)  # init 
    array = array.unstack()
        
    coords['time'] = time_coord
        
    return array

def metrics(x_test, y_test, lat, lon, coords, models, aux_data=None, eps=0.05):
    # models is a list containing trained instances of a model, collect statistics for this model 
    metrics = {}

    y_pred_list = evaluate_models(models, x_test, coords, lat, lon, take_mean=False, stack=True)
    
    # mask out points below elevation level 
    if aux_data is not None and False:
        y_test = unstack(y_test, coords)
        aux_data = np.resize(aux_data, (257, 241))
        elevation_mask =  aux_data > 4
        y_test = y_test.values[:, elevation_mask]
        nan_mask = ~np.isnan(y_test)
        y_test = y_test[nan_mask]
        y_pred_list = [unstack(y_pred, coords).values[:, elevation_mask][nan_mask] for y_pred in y_pred_list]

    metrics = [tf.reduce_mean(tf.keras.metrics.mean_squared_error(y_test, y)) for y in y_pred_list]

    return metrics

def signifcant_test(metric_dict):
    # perform wilcoxon significance test 
    # extracts entries for table 1 and 2, in the form of [mean, std, is_signifcant]
    
    def test_significance(x, y, alpha=0.05):
        _, p = wilcoxon(x, y)
        if p < alpha:
            return np.sign(np.mean(np.array(x) - np.array(y)))
        else:
            return False 
    
    new_metric = {loc: {} for loc in metric_dict[(0, 1.0)]}
    new_metric.update({qr: {} for qr in metric_dict if qr != (0, 1.0)})
    
    # extract entries from metric dict for table 1 and 2 
    for qr in metric_dict:
        qr_name, qr_list = None, None 
        for ln in metric_dict[qr]:
            ln_name, ln_list = None, None 
            
            # only entries from table 1 or 2 
            if qr != (0, 1.0) and ln != 'New Zealand':
                continue 
            
            first_mn = [mn for mn in list(metric_dict[qr][ln].keys()) if 'WeakLearner' not in mn][0]
            
            if qr == (0, 1.0):
                name, mse, key = ln_name, ln_list, ln
            else:
                name, mse, key = qr_name, qr_list, qr
            
            for n, (mn, mse_list) in enumerate(metric_dict[qr][ln].items()):
                new_metric[key][mn] = [np.mean(mse_list), np.std(mse_list), False]
                # dont test signifance of weak learner 
                if 'WeakLearner' in mn:
                    continue
                    
                # find the significant model, if any 
                if mse is None:
                    name = mn
                    mse = mse_list
                    continue
                    
                sig_result = test_significance(mse_list, mse)
                if sig_result == -1:
                    mse, name = mse_list, mn
                elif sig_result is False:
                    name = None
                
            # update 
            if qr == (0, 1.0):
                ln_name, ln_list = name, mse
            else:
                qr_name, qr_list = name, mse
                
            if ln_name is not None:
                new_metric[ln][ln_name][2] = True

        if qr_name is not None:
            new_metric[qr][qr_name][2] = True
            
    return new_metric

def collect_metrics(
                        models, # model dict, maps model_name -> [instances]
                        data_dict, 
                        Q_ranges=None, 
                        expr_name='default', 
                        loc_dict=None   
                   ):
    
    # creates the dataset that gives model performance, entries of this dataset depend on parameters, 
    # e.g. if auxiliary data should be considered within 
    
    metric_dict = {}
    y_test = data_dict['y_test']
    x_test = data_dict['x_test']
    coords = data_dict['coords']
    
    # populate a 3D dataset; 
    # Quantile range -> location -> model_name -> mean, std, True if signifcant, False otherwise, None if not applicable  
    for k, qr in enumerate(Q_ranges):
        metric_dict[qr] = {} 
            
        for n, (loc_name, (lat, lon)) in enumerate(loc_dict.items()): 
            # disaggregate data on spatial then quantile dimensions
            # so that if the quantile is taken for region, measure only rainfall of that region 
            
            # find data to test on based off disaggregations 
            segment_space = lat is not None and lon is not None
            do_qsplit = qr[0] != 0 or qr[1] != 1.0
            if segment_space and do_qsplit:
                yq = unstack(y_test, coords).sel(lat=lat, lon=lon).stack(z=['lat','lon']).dropna('z')  
                xq, yq = data_between(yq, qr[0], x_test, qr[1])
            elif segment_space:
                yq =  unstack(y_test, coords).sel(lat=lat, lon=lon).stack(z=['lat','lon']).dropna('z')  
                xq = x_test
            elif do_qsplit:
                xq, yq = data_between(y_test, qr[0], x_test, qr[1])
            else:
                xq, yq = x_test, y_test
            
            
            if loc_name not in metric_dict[qr]:
                metric_dict[qr][loc_name] = {}
                
            
            # evaluate model over test set 
            for model_name, model_list in models.items():
                    # _metrics is list of repetitions 
                    _metrics = metrics(
                                                          xq,
                                                          yq, 
                                                          lat,
                                                          lon,
                                                          data_dict['coords'],
                                                          model_list,
                                                          aux_data=data_dict['auxiliary'] if '>' in loc_name else None 
                                      )
                    
                    
                    # populate final entry so that the table methods
                    metric_dict[qr][loc_name][model_name] = [t.numpy() for t in _metrics]     
                    
                    
    
    metric_dict = signifcant_test(metric_dict)
    return metric_dict
