import numpy as np 
import matplotlib.pyplot as plt 
import xarray as xr
import os 

from useful_functions import * 


def load_histories(model_names=['base', 'avg', 'avgmax', 'max'], experiment_name='default', nmodels=1, baseline_dir=None):
    train_names = ['loss']
    epoch_names = ['loss', 'val_loss']
    
    
    # returns a dict/array for a model containing relevant statistics obtained during training/testing run. Load from specific file  
    def _data(
                  path,                                                   # Name of model 
                  files=['loss', 'tloss'],                                # Files where the statistics for model are stored 
                  cut=False,                                              # reduce size of input 
                  return_dict=True                                        # should output be dict or array 
              ):

        data_dict = {}
        for saved_file in files:
            d = np.loadtxt(f'{path}/{saved_file}')
            data_dict[saved_file] = d
            if cut:
                data_dict[saved_file] = d[min(len(d)-1, 0):]
            if not return_dict:
                data_dict = data_dict[saved_file]
                break
    
        return data_dict 
    
    # loader for history class, returns a dictionary where the model is mapped to the dictionary/list of statistics obtained from the run 
    def _dict(
                         data_files,
                         load,
                         experiment_name='default',
                         twod_dict=True, 
                         segment_data=False,
                         nmodels=1
              ):
        
        hist = {}
        for k in range(nmodels):
            for n, model_name in enumerate(load):
                # path
                if model_name == 'base':     
                    path = f'{baseline_dir}/base/{k}'
                elif experiment_name == 'quantile_alg':
                    path = f''
                else:
                    path = f''
                    
                data = _data(path, files=data_files, cut=segment_data, return_dict=twod_dict)
                
                if k == 0:
                    hist[model_name] = [data]
                else:
                    hist[model_name].append(data)
        
        return hist 
    
    
    # load dictionaries for training and testing statistics and predictions on test and trian test respectively. 
    train = _dict(train_names, load=model_names, experiment_name=experiment_name, nmodels=nmodels)
    epoch = _dict(epoch_names, load=model_names, experiment_name=experiment_name, nmodels=nmodels)
    
    return train, epoch


def save(history, folder):
    directory=f''
    if not os.path.exists(directory):
        os.mkdir(directory)
        
    for key, metric_log in history.items():
        np.savetxt(f'{directory}/{key}', metric_log)

def disaggregate(vector, batch_size=365//4):  # split up data in batches for readability  
    return np.array_split(vector, batch_size)

# reduce size, focus on particular location 
def segment(data, n):
    data = data.isel(time=slice(0, n * 365))  # reduce size of data temporally   
    
    return data 

def coarsen_input(data):
    return data.coarsen(lat=4, boundary='pad').mean().coarsen(lon=4, boundary='pad').mean()

def prep_rainfall(data, n):
    data = drop_nan_values(data)
    data = segment(data, n)
    coords = data.coords

    return data, coords

# interpoalte data to make it square 
def make_square(data):
    for im_set in data.channels:
        new_dim = max(im_set.lon, im_set.lat)
        im_set.interp(lon=new_dim, lat=new_dim)
    
    return data 

def prep_predictors(data, n, coarsen=False, do_square=False):
    data = segment(data, n)
    
    # pass in all predictors into train method
    data = data.sel(channel=['q_850', 't_850', 'w_850', 'u_850', 'v_850'])
    
    # coarsen predictands to simulate GCM input
    if coarsen:
        data = coarsen_input(data)
        
    # interpolate so that dimensions of inputs are equal 
    if do_square:
        data = make_square(data)

    return data


def prep_aux(aux_data):
    surrogate = drop_nan_values(aux_data)

    mean = np.mean(surrogate.values)
    std = np.std(surrogate.values)
    
    #thrs = 300 
    #aux_data.values = np.where(aux_data.values < thrs, thrs, aux_data.values)
    is_ocean = np.isnan(aux_data.values)
    aux_data.values = np.where(is_ocean, mean, aux_data.values) 
    aux_data.values = (aux_data.values - mean) / std
    aux_data.values = np.where(np.invert(is_ocean), aux_data.values + np.abs(np.min(aux_data.values)) + 1, aux_data.values)
    #aux_data.values = np.around(aux_data.values) 
    
    # expand dimensions to (1, 247, 257, 1)
    aux_data = aux_data.assign_coords(y='coord_value')
    aux_data = aux_data.expand_dims('y', axis=0)
    aux_data = aux_data.assign_coords(x='coord_value')
    aux_data = aux_data.expand_dims('x', axis=-1)
    
    return aux_data 
    

# loads and process the data 
def open_data():
    # load the data here 
    x_train = xr.open_dataset(r'/nesi/project/niwa03712/group_shared/x_train_25km.nc').w_200
    x_test = xr.open_dataset(r'/nesi/project/niwa03712/group_shared/x_test_25km.nc').w_200
    y_train = xr.open_dataset(r'/nesi/project/niwa03712/group_shared/y_train_25km.nc').Rain_bc
    y_test = xr.open_dataset(r'/nesi/project/niwa03712/group_shared/y_test_25km.nc').Rain_bc
    # elevation over domain 
    elevation = xr.open_dataset(r'/nesi/project/niwa03712/group_shared/vcsn_rainfall_augmented.nc').elevation
    
    # the amount of trianing and testing data in years (sample every day)
    N_train = 1
    N_test = 1
    
    # prepreocess the data 
    y_train, train_coords = prep_rainfall(y_train, N_train)
    y_test, test_coords = prep_rainfall(y_test, N_test)
    x_train = prep_predictors(x_train, N_train, coarsen=True) # coarsen input data to liken it to that of a GCM 
    x_test = prep_predictors(x_test, N_test, coarsen=True)
    elevation = prep_aux(elevation)
    
    # stack in this dict 
    data_dict = {
                    'x_train': x_train.values,
                    'x_test': x_test.values,
                    'y_train': y_train,
                    'y_test': y_test,
                    'auxiliary': elevation,
                    'coords': train_coords,
                }
    
    return data_dict