import copy
import numpy as np 
import tensorflow as tf 
import matplotlib as mpl
import os 
from Modules import * 
from ConvolutionalNetworks import * 
from data_handling import * 
from plot_functions import * 
from functools import partial 
import pandas as pd 
from sklearn.mixture import GaussianMixture
from new_baseline import empirical_mean_baseline
    
def load_models(    
                    names,                       # names of models to be loaded 
                    experiment_name,             # parent directory 
                    num_repeats=1,               # number of repititions to load 
                    num_models=1,                # number of weak learners if QB is loaded 
                    baseline_dir=None            # directory where models are stored 
               ):
    
    # to obtain QRE, load in weight network and weak learners 
    def load_qb(directory, CNN_dict, k):
        omega = tf.keras.models.load_model(f'{directory}/omega/{k}', compile=False)
        
        # change this line for fixed weight experiment 
        fixed_weights = [1/len(CNN_dict) for _ in CNN_dict] 
        
        return ynetwork(CNN_dict, omega, fixed_weights)
    
    directory = f''
    
    # this line wont trigger itself 
    CNN_dict = load_models([f'CNN_{n}' for n in range(num_models)], experiment_name, num_repeats) if 'qre' in names else None

    # load models in a dictionary, with key value pair model_name -> instances of model in a list
    models = {}
    for k in range(num_repeats):
        for n, sudo_path in enumerate(names):
            if sudo_path == 'base' and baseline_dir is not None:
                model = tf.keras.models.load_model(f'{baseline_dir}/base/{k}', compile=False)
            elif sudo_path == 'qre':
                model = load_qb(directory, CNN_dict, k)
            else:
                model = tf.keras.models.load_model(f'{directory}/{sudo_path}/{k}', compile=False)
            
            # wrap BGNet model 
            model = BGCallWrapper(model) if sudo_path != 'mse' else model
            if k == 0:
                models[sudo_path] = [model]
            else:
                models[sudo_path].append(model)
    
    return models


# plots training history for each of the CNNs from the quantile alg in a seperate window 
def call_CNN_plot(num_models, epoch_hist, repeats, directory):
    for n in range(num_models):
        name = f'CNN_{n}'
        sub_dict = {name: epoch_hist[name]}
        plot_epoch_history(sub_dict, repeats, directory)
        

def eval_pipeline(
                        models,                                        # dictionary of model instances 
                        data_dict,                                     # data dictionary 
                        save_dir,
                        Q_ranges=None,                                 # quantile intervals to evaluate over
                        use_aux=False,                                 # if NZ+ is included 
                        loc_dict=None,                                 # locations to evaluate over 
                ):
    
    # put elevation data to be a vector of the same dimension as precipitation samples, with a one-to-one corrrespondence between grid points
    if use_aux:
        aux_data = xr.where(data_dict['auxiliary'] == 0.0, np.nan, data_dict['auxiliary'])
        aux_data = drop_nan_values(data_dict['auxiliary'])
        aux_data = np.squeeze(aux_data)
    else:
        aux_data = None 
    
    # add these out-of-domain intervals for evaluation
    Q_ranges.insert(0, (0, 1.0))
    Q_ranges.insert(0, (0, 0.2))
    Q_ranges.insert(0, (0.9, 1.0))
    
    # collect result dict 
    metric_dict = collect_metrics(models, data_dict, Q_ranges, loc_dict=loc_dict)
    print(metric_dict)


def get_baseline(odim, data_dict): # returns a BGNet(-) instance 
    return BGNet(
                                odim,
                                coords=data_dict['coords'],
                                baseline=True,
                )


def get_bgnet(odim, data_dict, use_aux=False, num_dense=[256]):   # returns a BGNet instance 
    return BGNet(
                                    odim,
                                    coords=data_dict['coords'],
                                    baseline=False,
                                    ch_attn=True,
                                    avg_pool=True,
                                    max_pool=True,
                                    sp_attn=False,
                                    kernel_sizes=[6, 6, 6],
                                    aux_data=data_dict['auxiliary'],
                                    n=data_dict['x_train'].shape[1],
                                    m=data_dict['x_train'].shape[2]
                ) 


def my_compile(model, loss=None, schedule=None, metrics=None): 
    # returns a compiled tensorflow model 
    
    if schedule is None:
        schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                                    initial_learning_rate=10**-4,
                                                                    decay_steps=6000,
                                                                    decay_rate=0.7
                                                              )    
    
    
    
    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=schedule), 
                        loss=GammaLoss() if loss is None else loss,
                        metrics=[BGStatWrapper('mse', tf.keras.metrics.MeanSquaredError())] if metrics is None else metrics
                 ) 
   

def train_on_quantiles(
                            data_dict,                       # data dictionary 
                            Q_ranges,                        # quantile intervals to split over
                            num_repeats=1,                   # number of repeats 
                            expr_name='quantile_alg',        # directory to save in 
                            model_factory=get_bgnet,         # obtains a instance of a weak learner for QB, default is BGNet
                            do_save=True                     # determines if models are saved 
                       ):
    
    
    # trains the weak learners on segmentations of the dataset 
    
    assert Q_ranges[0][0] == 0 and Q_ranges[-1][-1] == 1
    
    output_dim = data_dict['y_train'].shape[-1]
    phist, ehist, models = {}, {}, {}
        
        
    # a loop for creating the weak learner and its training data 
    for n, (q_low, q_high) in enumerate(Q_ranges):
        # perform segmentation 
        Xq, yq = data_between(data_dict['y_train'], q_low, data_dict['x_train'], q_high)
        data = {'x_train': Xq, 'y_train': yq}        
        
        cnn_model_fac = {f'WeakLearner{n}': lambda: model_factory(output_dim, data_dict, False)}
        
        # train weak learner on the appropriate segementation of data, add results of weak learner to existing results  
        phist, ehist, models = concat_dict(
                                                collect_experiment_data(
                                                                           cnn_model_fac, 
                                                                           data, 
                                                                           expr_name, 
                                                                           num_repeats, 
                    schedule=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=10**-4, decay_steps=6000, decay_rate=0.7),
                                                                           save_models=do_save,
                                                                           pat=15
                                                                       ),
                                                [phist, ehist, models]
                                          )
                                           
            
                
    return phist, ehist, models   


def quantile_alg(
                    data_dict,
                    cnn_dir,                                 
                    ynet_dir=None, 
                    load_cnn=True,
                    cnst_base=False,
                    num_repeats=1,
                    Q_ranges=[(0, 0.2), (0.2, 1.0)],
                    expr_name='aux_attn',
                    do_save=True
                ):
    
    # returns the ensemble of the Quantile-Boost algorithm 
    
    n = data_dict['x_train'].shape[1]
    m = data_dict['x_train'].shape[2]
    
    # if trainig all cnns not nessacary, e.g. when testing only omega, then load them in from memory 
    if load_cnn:
        CNN_dict = load_models(
                                 [f'CNN_{n}' for n in range(num_models)], 
                                 expr_name, 
                                 num_repeats, 
                              )
        phist, ehist = {}, {}
    elif cnst_base:
        CNN_dict = empirical_mean_baseline(data_dict['y_train'], Q_ranges, num_repeats)
        phist, ehist = {}, {}
    else:
        # train weak learners 
        phist, ehist, CNN_dict = train_on_quantiles(data_dict, Q_ranges, num_repeats,
                                                    expr_name=expr_name, do_save=do_save)
        
    # shallow copy, just reference key -> value pair 
    models = CNN_dict.copy()
    
    # buld the discretisbed dataset
    
    num_bins = len(Q_ranges)
    quantiles = bin_y_var(data_dict['y_train'], Q_ranges)
    data = {'x_train': data_dict['x_train'], 'y_train': xr.DataArray(quantiles)}
        
    # train omega (Weight Network)
    phist, ehist, models = concat_dict(
        
                                           collect_experiment_data(
                                                                       {'omega': lambda: Omega(num_bins)}, 
                                                                       data, 
                                                                       expr_name, 
                                                                       num_repeats, 
                                                                       losses={'omega': OmegaLoss(num_bins)}, 
                              schedule=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=5*10**-5, decay_steps=6000, decay_rate=0.8),  
                                                                       metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.MeanAbsoluteError()],
                                                                       pat=4,
                                                                       save_models=False
                                                                   ),
                                          [phist, ehist, models]
                                       )
    
    # put omega and CNN models together for the ynet model  
    # includes the fixed weight methods denoted in the paper as Probability (pqb) and Bagging (cqb)
    models['qre'] = [] 
    cnst_weights = [1/len(CNN_dict) for _ in CNN_dict]
    prob_weights = [q2 - q1 for q1, q2 in Q_ranges]
    for omega in models['omega']:        
        y_net = ynetwork(CNN_dict, omega)
        models['qre'].append(y_net)
    
    del models['omega']
    
    # collect fixed weight methods 
    models['cqre'] = []
    models['pqre'] = [] 
    bag_weights = [1/len(CNN_dict) for _ in CNN_dict]
    prob_weights = [q2 - q1 for q1, q2 in Q_ranges]
    for k in range(num_repeats):
        cnn_subdict = {}
        for n, cnn_list in enumerate(CNN_dict.values()):
            cnn_subdict[f'CNN_{n}'] = [cnn_list[k]]

        models['cqre'].append(ynetwork(cnn_subdict, fixed_weights=cnst_weights))
        models['pqre'].append(ynetwork(cnn_subdict, fixed_weights=prob_weights))
    
    del models['cqre']
    del models['pqre']
        
    return phist, ehist, models, CNN_dict


def set_up_experiment(
                        data_dict,                  # data 
                        save_dir,                   # parent directory 
                        num_repeats                 # number of repititions 
                    ):
    
    # run experiments for the base models, speficy the models to run in the dictionary below 
    
    output_dim = data_dict['y_train'].shape[1]
    n = data_dict['x_train'].shape[1]
    m = data_dict['x_train'].shape[2]
        
    # BGNet(-)
    base = lambda: BGNet(
                                        output_dim,
                                        coords=data_dict['coords'],
                                        baseline=True,
                                        ch_attn=False,
                                        avg_pool=False,
                                        max_pool=False,
                                        sp_attn=False,
                                        num_channels=[64, 128, 256],
                         )
    
    # BGNet(-) + aux data 
    aux = lambda: BGNet(
                                        output_dim,
                                        coords=data_dict['coords'],
                                        aux_data=data_dict['auxiliary'],
                                        baseline=True,
                                        ch_attn=False,
                                        sp_attn=False,
                                        n=n,
                                        m=m
                        )
    
    
    # Embed as in table 1 
    embed_ch = lambda: BGNet(
                                    output_dim,
                                    coords=data_dict['coords'],
                                    baseline=False,
                                    ch_attn=True,
                                    avg_pool=True,
                                    max_pool=True,
                                    sp_attn=False,
                                    aux_data=data_dict['auxiliary'],
                                    kernel_sizes=[6, 6, 6],
                                    n=n,
                                    m=m
                            ) 
    
    # interpolate as in table 1
    interp_ch = lambda: BGNet(
                                    output_dim,
                                    coords=data_dict['coords'],
                                    baseline=False,
                                    ch_attn=True,
                                    avg_pool=True,
                                    max_pool=True,
                                    sp_attn=False,
                                    aux_data=data_dict['auxiliary'],
                                    kernel_sizes=[6, 6, 6],
                                    n=n,
                                    m=m,
                                    aux_mode='interp'
                            ) 
    

    # BGNet
    ch = lambda: BGNet(
                                    output_dim,
                                    coords=data_dict['coords'],
                                    baseline=False,
                                    ch_attn=True,
                                    avg_pool=True,
                                    max_pool=True,
                                    sp_attn=False,
                                    kernel_sizes=[6, 6, 6],
                          ) 
    
    # BGNet with spatial attention after channel attention, as in woo 2019 
    aux_sptch = lambda: BGNet(
                                    output_dim,
                                    coords=data_dict['coords'],
                                    baseline=False,
                                    ch_attn=True,
                                    avg_pool=True,
                                    max_pool=True,
                                    sp_attn=True,
                                    kernel_sizes=[6, 6, 6],
                                    aux_data=data_dict['auxiliary'],
                                    n=n,
                                    m=m
                          ) 
    
    # the model trained on MSE, same as BGNet(-) but only one dense at end 
    mse = lambda: MSENet(output_dim, data_dict['coords'])
    
    
    # add or remove models from this dictionary to run them in an experiment 
    model_facs = {
                     #'sptch': sptch,
                     #'aux_sptch': aux_sptch, 
                     #'aux': aux,
                     #'embedch': embed_ch, 
                     #'interpch': interp_ch,
                     'BGNet': ch,   
                     #'BGNet(-)': base,  
                     #'MSENet': mse
                     
                 }
    
    # speficy losses here 
    losses = {name: GammaLoss() if name != 'mse' else tf.keras.losses.MeanSquaredError() for name in model_facs}
    
    # run experiment 
    performance_stats, epoch_hist, models = collect_experiment_data(
                                                                        model_facs, 
                                                                        data_dict, 
                                                                        save_dir, 
                                                                        num_repeats,
                                                                        save_models=False,
                                                                        losses=losses,
                                                                        pat=15
                                                                   )
    
    return performance_stats, epoch_hist, models 

def exe_run(
                    model,                # a compile tensorflow model 
                    _data,                # data to run 
                    sudo_path,            # name of model, what to save it as
                    save_model=True,      # decides if model should be saved 
                    _pat=4                # the patience of early stopping. Weight network uses 4 for the classification task, BGNet uses 14
            ):
    
    
    model_path = f''    # path in directory to model 
    bs = 365//4                                                           # approximately a season
    num_epochs = 400                                                   # overtrain epochs but use early stopping 

    # callbacks 
    epoch_logger = EpochLogger()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=_pat, restore_best_weights=True)
    
    # train and potentially save model 
    with tf.device('GPU:0'):
        epoch_history = model.fit(
                                        _data['x_train'],
                                        _data['y_train'].to_numpy(), 
                                        batch_size=bs, 
                                        callbacks=[early_stop, epoch_logger], 
                                        epochs=num_epochs, 
                                        validation_split=0.2
                                  ).history

        if save_model:
            model.save(model_path)
        ms = str(model.summary())
    
    # statistics over last epoch
    performance_stats = epoch_logger.get_statistics()
    
    return performance_stats, epoch_history

def collect_experiment_data(
                                model_factories,               # a dictionary of model_name -> function which initialises model 
                                data_dict,                     # data 
                                save_dir,                      # parent directory to save model under 
                                num_repeats,                   # repititions of experiment 
                                losses=None,                   # losses to use for model training 
                                schedule=None,                 # learning rate schedule, default exp decay is used through out 
                                metrics=None,                  # what metric to track for model 
                                save_models=True,              # should model be saved 
                                pat=4                          # paitence level 
                            ):
    
    
    # trains the model and returns the run results in the form of dictionary with key-value pairs model name -> [instances, ...]
    # compile model based off parameters 
    
    pstat, ehist, models = {}, {}, {}
    
    # train each model for the set number of repetitions
    for model_name, model_factory in model_factories.items():
        
        pstat[model_name] = []
        ehist[model_name] = [] 
        models[model_name] = [] 
        
        for k in range(num_repeats):
            # use factory to init a model 
            model = model_factory()  
            my_compile(model, losses[model_name] if losses is not None else losses, schedule, metrics)
            
            ps, eh = exe_run(model, data_dict, f'{save_dir}/models/{model_name}/{k}', save_models, _pat=pat)
            model = BGCallWrapper(model)
            
            pstat[model_name].append(ps) 
            ehist[model_name].append(eh)
            models[model_name].append(model)
            
    
    return pstat, ehist, models


def call_unstack(data_dict):
    data_dict['y_train'] = unstack(data_dict['y_train'], data_dict['coords'])
    data_dict['y_test'] = unstack(data_dict['y_test'], data_dict['coords'])

def meta_data():
    # meta data used to configure the experiments
    _meta_data = {

        
                        # determines if models are to be loaded in from memory
                        'load': False, 
        
                        # if enabled, load weak learners and then train the Weight Network        
                        'qa_load_cnn': False, 
        
                        # determines if models are saved after a run, note that saving and loading requrie sub directories to be made   
                        'save_models': False, 
        
                        # parent directory to save models and figures to 
                        'save dir': '',
        
                        # list of experiments to be run, two possible options; "standard" runs base models and 'quantile alg' runs Quantile Boost
                        'expr type': ['quantile alg'],     
                        
                        # the number of segmentations to do, N in paper 
                        'n_models': 2,
                        
                        # models to load if this option was enabled 
                        'model names': ['ch', 'base'],
        
                        # how many times to repeat each experiment 
                        'repeats': 1
                }
    
    
    return _meta_data 

# get quantile intervals 
# perform gmm clustering over sum of each sample 
def get_Q_ranges(y_train, nmodels):
    N = y_train.shape[0]
    sum_y = np.expand_dims(np.sum(y_train, axis=-1), axis=-1)
    gmm = GaussianMixture(n_components=nmodels, n_init=100).fit(sum_y) 
    counts = np.sort([np.sum(np.where(sum_y <= y_upper, 1.0, 0.0)) for y_upper in gmm.means_])
    
    means = gmm.means_
    covariances = gmm.covariances_
    
    Q_ranges = [c/N for c in counts]
    # create quantile intervals 
    Q_ranges.insert(0, 0)
    Q_ranges[-1] = 1.0   # expand largest quantile to end point 
    Q_ranges = [(Q_ranges[n], Q_ranges[n+1]) for n in range(len(Q_ranges) - 1)] 
    
    return Q_ranges

def run_file():
    meta = meta_data()
    
    # loads in data, which has the form {'x_train': xr.DataArray, 'y_train': np.array, 'x_test': xr.DataArray, 'y_test': np.array, 
    # 'coords': the xr.DataArray.coords of the data array y variables orginally are 257 x 256, but after removing values outside of domain, it       becomes a vector     # of shape 11471, the coord variable is to reverse this process
    
    data_dict = open_data()    

    Q_ranges = get_Q_ranges(data_dict['y_train'], meta['n_models'])
    
    save_dir = meta['save dir']
    
    # in the case the models have been saved 
    if meta['load']:
        models = load_models(
                                 meta['model names'], 
                                 meta['name'], 
                                 meta['repeats'], 
                                 meta['nmodels'],
                                 baseline_dir=f'{save_dir}/models' # path to model save folder
                            )
        
        
        train_hists, epoch_hist = load_histories(
                                                    meta['model names'],    
                                                    meta['name'],
                                                    nmodels=meta['repeats'], 
                                                    baseline_dir=f'{save_dir}/runs'
                                                 )
                                      
    # will run whichever experiment is specfied in the meta data, concatenates results together 
    else:
        use_standard = 'standard' in meta['expr type']
        use_quantile = 'quantile alg' in meta['expr type']
        tran_hist, epoch_hist, models = {}, {}, {}
        if use_standard:
            _, _, standard_models = set_up_experiment(data_dict, save_dir, meta['repeats']) # now extra dimension of num repeats
        if use_quantile:
            _, _, quantile_models, CNN_list = quantile_alg(
                                                                           data_dict,
                                                                           f'{save_dir}/models',            
                                                                           load_cnn=meta['qa_load_cnn'],
                                                                           num_repeats=meta['repeats'],
                                                                           cnst_base=True,  # if Naive-Ensemble 
                                                                           Q_ranges=Q_ranges,
                                                                           do_save=meta['save_models']
                                                          )  
        # concate dictionaries together 
        models = {}
        if use_quantile and use_standard:
            models = {**quantile_models, **standard_models}
        elif use_quantile:
            models = quantile_models
        elif use_standard:
            models = standard_models
        else:
            raise ValueError('Incorrect experiment types')
    
    # dictionary for areas to evaluate models over, slices are because data is a xarray 
    # None, None is entire NZ, New Zealand(>4) is grid points of greater elevation than 1305m on the unnormalised data 
    loc_dict = {
                    'New Zealand': [None, None],
                    'New Zealand(>4)': [None, None],
                    'Auckland': [slice(-37, -36), slice(174, 175)], 
                    'Wellington': [slice(-41.8, -40.8), slice(174, 175)],
                    'Napier': [slice(-40, -39), slice(176, 177)],
                    'Milford Sound': [slice(-45, -44), slice(167, 168)],
                    'Mt Cook': [slice(-44, -43), slice(170, 171)],
                    'Grey Mouth': [slice(-43, -42), slice(171, 172)],
                    'Tongariro': [slice(-40, -39), slice(175, 176)],
               }
    
    # evaluate the models on the test set, across both quantile intervals and locations and perform wilcoxon signifcance test
    # generates both table 1 and 2 
    test_result = eval_pipeline(models, data_dict, save_dir, Q_ranges, loc_dict=loc_dict, use_aux=False)
    
    cmaps = ['Reds', 'cividis']
    
    def mse(x, y):
        return np.mean(np.square(x - y), keepdims=True, axis=0)

    def mse90(y_true, y_pred):
        qsp = np.linspace(0.9, 1.0, 10)
        y = np.quantile(y_true, qsp, axis=0, keepdims=False)
        x = np.quantile(y_pred, qsp, axis=0, keepdims=False)
        return mse(x, y)

    funcs = {'MSE90': mse90, 'MSE': mse}
    
    """QQ_plot(   
                    models, 
                    data_dict['y_test'], 
                    data_dict['x_test'], 
                    data_dict['coords'], 
                    loc_dict, 
                    directory='/home/bailiet/QB/slurm',
                    color_list=['#67597A', '#85BAA1', 'red']
            )
    
    plot_model_output(
                            models, 
                            data_dict['x_test'], 
                            data_dict['y_test'], 
                            data_dict['coords'], 
                            meta_data,
                            funcs,
                            cmaps,
                            '/home/bailiet/QB/slurm'
                      )"""

run_file()