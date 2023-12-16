import numpy as np
import matplotlib as mlp 
import matplotlib.animation as animation
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable 
import pandas as pd 

from scipy import ndimage
from useful_functions import * 
from data_handling import *

import os 
import imageio 
import cartopy.crs as ccrs 


# plots stat map over quantile of rainfall 
def plot_statq(
                    models,
                    y_true,
                    ximages,
                    coords,
                    name,
                    Q_ranges=[(0, 0.4), (0.4, 0.8), (0.8, 1.0)],
                    stat_map=MSE,
                    lat=None,
                    lon=None,
                    q_step=0.1,
                    directory=''
              ):

    stat_list = []
    for model_name, model in models.items():
        for (q_low, q_high) in Q_ranges:
            Xq, yq = data_between(ximages, y_true, q_low, q_high)
            y_pred = evaluate_models(model, Xq, lat=None, lon=None)
            #print('Y PRED SHAPE', y_pred.shape)
            #print('Y TRUE SHAPE', y_true.shape)
            #stat = stat_map(y_true, y_pred)
            #stat_list.append(stat)
    
    #fig, axs = plt.subplots()
    #axs.plot(stat_list)
    #plt.close(fig)
        
        
def plot_data(
                    ximages,
                    yimages,
                    q_step=0.01,
                    Q_ranges=[(0.0, 0.5), (0.5, 1.0)],
                    colors=['red', 'blue'],
                    directory='',
                    name='y_data'
             ):
    
    # Qspace is plot space, Q_ranges is color ranges 
    Q_space = np.arange(0, 1 + q_step, q_step)
    y_prob = []  # probability y is in the bin Q(i - 1), Q(i))
    x_prob = []
    y_sum = []
    x_sums = [[] for _ in range(ximages.shape[-1])]
    N = yimages.shape[0]
    for n, q_high in enumerate(Q_space):
        q_low = Q_space[n - 1]
        xq, yq = data_between(yimages, q_low, ximages, q_high=q_high)
        
        if xq.shape[0] == 0:
            continue
        
        p = yq.shape[0] / N
        y_prob.append(p)
        x_prob.append(xq.shape[0] / N)
        y_sum.append(np.sum(yq))
        
        for i in range(ximages.shape[-1]):
            x_sums[i].append(np.sum(np.abs(xq[:,:,:,i])))
                           

    #axs.plot(y_sum, y_prob, color='black', linewidth=2)
    for n, sum_array in enumerate([y_sum]+x_sums):
        plt.figure(figsize=(10, 10))
        # color graph, draw lines 
        N = len(y_prob)
        for k, (ql, qh) in enumerate(Q_ranges):
            upperk = min(int(np.floor(N * qh)), N - 1)
            lowerk = min(int(np.floor(N * ql)), N - 1)
            if qh != 1.0:
                plt.vlines(qh, 0, sum_array[upperk], color='black', linewidth=2)
            graph = sum_array[lowerk:upperk]
            segment = np.linspace(ql, qh, len(graph))
            plt.plot(segment, graph, color='black', linewidth=2)
            plt.fill_between(segment, graph, color=colors[k], alpha=0.3)
            
        plt.ylim(0, np.max(sum_array))
        plt.xlim(0, 1)
        plt.xticks([0.0, 1.0], fontsize=15)
        plt.yticks([min(sum_array), max(sum_array)], fontsize=15)
        plt.savefig(f'{directory}/prob_{n}.png', transparent=True)
        plt.close() 
   


def QQ_plot(
                models, 
                y_true, 
                ximages, 
                coords, 
                location_dict, 
                directory='',
                name='QQ',
                color_list=['#85BAA1', '#67597A', 'red']
            ):
    
    def segment(x, lat, lon):
        x = unstack(x, coords)
        if lat is not None and lon is not None:
            x = x.sel(lat=lat, lon=lon)
        x = drop_nan_values(x, dim0='lat', dim1='lon')
        
        return x
    
    def summations(predictand, q_space, lat=None, lon=None, thrs=2):
        predictand = segment(predictand, lat, lon)
        predictnad = predictand[predictand > 1]
        predictand = [np.quantile(predictand, q, axis=0) for q in q_space]
        predictand = np.array([np.mean(p) for p in predictand]) # some points will have zero shape 
        
        return predictand 
      
    q_space = np.linspace(0, 1, 20)
    eps = 0.8
    
    for save_name, (lat, lon) in location_dict.items():
        plt.figure()
        gt_summation = summations(y_true, q_space, lat, lon)
        for n, (model_name, _model) in enumerate(models.items()):
            y_pred_list = evaluate_models(_model, ximages, coords, None, None, take_mean=False)
            summation_list = [summations(y_pred, q_space, lat, lon) for y_pred in y_pred_list]
            mean = np.mean(summation_list, axis=0)
            std = np.std(summation_list, axis=0)

            gt_summation[-1] -= eps
            color = color_list[n%len(color_list)]
            plt.plot(gt_summation, mean, label=model_name, marker='o', markersize=5, linewidth=2, color=color)
            gt_summation[-1] += eps 
            plt.fill_between(gt_summation, mean - std, mean + std, alpha=0.25, color=color)
    
        plt.plot(gt_summation, gt_summation, label='Real', linestyle='--', linewidth=4, color='#646A6C')
        plt.xlabel(r'$\mathbf{Real}$', fontsize=12)
        plt.ylabel(r'$\mathbf{Prediction}$', fontsize=12)
        plt.xlim(0, gt_summation[-1] + eps)
        plt.ylim(0, gt_summation[-1] + eps)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.title(name, fontsize=14)
        plt.legend(edgecolor='black', fancybox=False)
        plt.savefig(f'{directory}/QQ_{save_name}.pdf')
        plt.close()

    
def box_plot(
                            models, 
                            ximages,
                            yimages, 
                            coords,
                            meta_data,
                            directory='',
                            color_list=['#67597A', '#85BAA1', 'red']
            ):
    
    # init an empty dict for the boxes, func name to statistic mapping 
    func_name = list(meta_data['funcs'].keys())[0]
    box_stats = {}
    for k in meta_data['funcs'].keys():
        if k == 'Real':
            continue
        box_stats[k] = []
        
    # first collect statistics, stat_name : model_stats 
    for k, (fname, fdata) in enumerate(meta_data['funcs'].items()):
        fmap = fdata['map']      
        for n, (model_name, model) in enumerate(models.items()):
            # collection of model predicitons 
            y_pred_list = evaluate_models(model, ximages, coords, take_mean=False)
         
            y_stat_list = [fmap(yimages, y_pred) for y_pred in y_pred_list]
            y_stat = xarray_mean(y_stat_list)
            box_stats[fname].append(y_stat)

    # now that data has been collected for the boxplots, we can plot them 
    model_names = list(models.keys())
    stat_names = list(box_stats.keys())
    
    # box positions, clusters have wider spacing than within clusters 
    box_dist = 0.65
    cluster_dist = 1.25
    positions = [[(idx + 1) * cluster_dist + (idy + idx) * box_dist for idy in range(len(stat_names))] for idx in range(len(model_names))]
    box_plots = [] # for legend 
    fig, axs = plt.subplots()
    for box_positions, (stat_name, stat_list) in zip(positions, box_stats.items()):    
        # Coloring and styling the boxes
        for n, (sn, pos, model_stat, color) in enumerate(zip(stat_names, box_positions, stat_list, color_list)):
            bp = axs.boxplot(
                                model_stat.stack(z=['lat', 'lon']).dropna('z'), 
                                patch_artist=True, 
                                positions=[pos], 
                                widths=0.4, 
                                showfliers=False, 
                                boxprops=dict(facecolor=color, linewidth=2),
                                whiskerprops=dict(color='black', linewidth=1),
                                capprops=dict(color='black', linewidth=1),
                                medianprops=dict(color='black', linewidth=2)
                            )
            
            # for legend 
            if len(positions[0])> len(box_plots):
                box_plots.append(bp)
            
    convert = {'base': 'Baseline', 'tch': 'Train+Attn', 'ch': 'Attn'}  # convert model names to more readable format 
            
    #axs.legend([convert[mn] for mn in model_names], loc='upper center', bbox_to_anchor=(0.5,1.1),  ncol=len(model_names), frameon=False)
    axs.legend([bp['boxes'][0] for bp in box_plots], model_names, loc='upper center', bbox_to_anchor=(0.5,1.1),  ncol=len(model_names), frameon=False)
    axs.tick_params(axis='y', labelsize=11)
    axs.set_xticks([np.mean(box_position) for box_position in positions])
    axs.set_xticklabels([r'$\mathbf{' + sn + r'}$' if sn != 'Real' else None for sn in stat_names], fontsize=12)
    axs.set_ylabel(r'$\mathbf{Statistic}$ $\mathbf{Value}$', fontsize=12)
    fig.suptitle(r'$\mathbf{Spatial}$ $\mathbf{Distributions}$', fontsize=14, y=0.99)
    fig.savefig(f'{directory}/box_plot.pdf')
                 

def plot_input_data(
                        data,
                        directory = ''
                    ):

    npredictors = data['x_train'].shape[-1]
    
    # model input plots
    for k in range(npredictors):
        predictor = data['x_train'].isel(channel=k, time=0)
        plt.figure()
        predictor.plot()
        #xr.plot.pcolormesh(predictors)
        plt.savefig(f'{directory}/model_input {k+1}.pdf')
        plt.close()

    plt.figure()
    data['auxiliary'].plot()
    plt.savefig(f'{directory}/elevation_data_processed.pdf')
    plt.close()
        
        
# plots training statistics, like loss, PSNR...  
def plot_train_stats(
                                  train_metrics, 
                                  models,
                                  data,
                                  experiment_name,
                                  directory = ''
                     ):    
    
    # a subplot for each stat monitored 
    mn = list(models.keys())[0] # arbitary key (model name)
    stat_names = list(train_metrics[mn][0].keys())
    num_stats = len(train_metrics[mn])
    means = [None for _ in range(num_stats)] 
    
    for model_name, train_array in train_metrics.items():
        fig_temp, axs_temp = plt.subplots()
        for train_data in train_array: # train_arry = [model#1, model#2, ...]
            for k, (stat_name, stat) in enumerate(train_data.items()):
                axs_temp.plot(stat) # plot solo figure 
                
                if k == len(train_data) - 1:
                    fig_temp.suptitle(f'Model {model_name.capitalize()} for statistic {stat_name.capitalize()}')
                    fig_temp.savefig(f'{directory}/{model_name}_{stat_name}.pdf')
                
                # get mean of stat k for model 
                if means[k] is None:
                    means[k] = stat
                else:
                    means[k] += stat
                
        axs_temp.set_xlabel('#Batch')
        axs_temp.set_ylabel('Loss')
        plt.close(fig_temp)
    
    
    # plot mean of each stat for every model on one figure 
    plt.figure()
    for stat_mean in means:
        plt.xlabel('#Batch')
        plt.ylabel('Loss')
        plt.plot(np.divide(stat_mean, num_stats))
        
    plt.close()
        
           
def concat_gifs(names, directory, gif_name='abc', nframes=10, fps=0.5):
    local_dir = f''
    
    # collect all gif objects
    gifs = []
    for name in names:
        gifs.append(imageio.mimread(f'{local_dir}/{name}.gif'))  # sometimes zero 
    gifs = np.array(gifs)
    
    # horiziontally stack frames of gif objects into array
    frames = []
    for n in range(nframes):
        images = []
        for k, gif in enumerate(gifs):
            images.append(gif[n])
        
        frames.append(np.hstack(images))
        
    imageio.mimsave(f'{directory}/{gif_name}.gif', frames, fps=fps)
    
    
def get_frames(local_dir, num_frames):
    frames = []
    for n in range(num_frames):
        frames.append(imageio.v2.imread(f'{local_dir}/#{n}.jpg'))
        
    return frames     
             
    
def save_to_local(
                        ximages,
                        yimages,
                        coords,
                        models, 
                        local_dir,
                        model_name,
                        num_images, 
                        fdata,
                        plot_name
                 ):
    
        fmap = fdata['map']
        color = fdata['color']
        vmin, vmax = fdata['vvals']    
        
        if not os.path.exists(local_dir):
            os.mkdir(local_dir)
                 
        for n, (input_im, output_im) in enumerate(zip(ximages, yimages)):
            if n > num_images:
                break
            
            input_im = np.expand_dims(input_im.to_numpy(), axis=0)
            
            y_pred_list = evaluate_models(models, input_im, coords, take_mean=False)
            
            plt_stat_list = [fmap(y_pred, output_im) for y_pred in y_pred_list]
            plt_stat = xarray_mean(plt_stat_list).squeeze()
            plt_stat.plot(add_colorbar=False)
            
            xr.plot.pcolormesh(plt_stat, cmap=color, vmin=vmin, vmax=vmax)
            plt.title(plot_name)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.savefig(f'{local_dir}/#{n}.jpg')
            plt.close()
            
    
def create_animations(
                            models,
                            x_data,
                            y_data,
                            coords,
                            meta_data, 
                            name='',
                            directory = f''
                      ):    
    
    for fname, fdata in meta_data['funcs'].items():
        if fdata['map'] is None:
            continue
            
        # first loop: iterate over the models # second: get the charcteristic (prediction, real data, difference between, ...)            
        for model_name, model in models.items():  
            local_dir = f'{directory}/{fname}'
            plot_name = fname if fname == 'Real' else model_name
            save_name = fname if fname == 'Real' else f'{model_name}_{fname}'
            save_to_local(
                                x_data, 
                                y_data, 
                                coords, 
                                model,         
                                local_dir, 
                                model_name, 
                                meta_data['nframes'],
                                fdata,
                                plot_name
                            )
            frames = get_frames(local_dir, meta_data['nframes'])
            imageio.mimsave(f'{directory}/{save_name}_{name}.gif', frames, fps=meta_data['fps']) # store gif as stand alone animation 

        
def plot_model_output(
                            models, 
                            ximages, 
                            yimages, 
                            coords, 
                            meta_data,
                            funcs,
                            cmaps,
                            directory
                     ):
    
    def get_ax(ax, N, n):
        if N == 1:
            return axs 
        else:
            return axs[n]
    
    model_names = list(models.keys())
    stat_names = list(funcs.keys())
        
    for k, (fname, fmap) in enumerate(funcs.items()):
        fig, axs = plt.subplots(1, len(models)+1, subplot_kw={'projection': ccrs.PlateCarree()}) # generate grid of plots     
        plt.subplots_adjust(wspace=0.75)
        y_stat, last_stat = None, None  
                
        cmap = cmaps[k % len(cmaps)]

        for n, (mn, model) in enumerate(models.items()):
            # get stat
            y_pred = np.stack(evaluate_models(model, ximages, coords, take_mean=False))[0]
            y_stat = fmap(y_pred, yimages)
            y_stat = unstack(y_stat, coords)
            last_stat = y_stat if last_stat is None else last_stat 
            
            ax = get_ax(axs, len(model_names), n)
            
            # plot into approiate grid cell 
            add_cbar = (n == len(models) - 1)
            cbar_ax = y_stat.plot(add_colorbar=True if add_cbar else False, ax=ax, cmap=cmap, cbar_kwargs={'shrink': 0.5} if add_cbar else None, add_labels=False)
            
        # None         
        ax = get_ax(axs, len(models), -1)
        y_res = y_stat - last_stat 
        y_res.plot(add_colorbar=True, ax=ax, cmap='RdBu', cbar_kwargs={'shrink': 0.5}, add_labels=False)
        last_stat = None 

        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_frame_on(False)
            ax.coastlines('10m')
    
        fig.savefig(f'{directory}/{fname}_spatial_plots.pdf')
    

# plot location, stat, model and quantiles all in a table 
def table(meta_result, directory=''):
    qr_ranges = list(meta_result.keys())
    stat_names = list(meta_result[qr_ranges[0]].keys())
    loc_names = list(meta_result[qr_ranges[0]][stat_names[0]].keys())
    model_names = list(meta_result[qr_ranges[0]][stat_names[0]][loc_names[0]].keys())
        
    table = r'\documentclass{article}'+'\n'+r'\usepackage{booktabs}'+'\n'+r'\usepackage{multirow}'+'\n'+r'\usepackage{siunitx}'+'\n'+r'\begin{document}'+'\n'
    table += r'\begin{table}'+'\n'+r'\begin{tabular}{llSSSS}'+'\n'+r'\toprule'+'\n'+r'\multirow{2}{*}{QR} & \multicolumn{2}{c}{Metric} & \multicolumn{' + f'{len(model_names)}' + r'}{c}{Method}\\'
    
    
    latex_name = {'base': 'Baseline', 'ch': 'BG-Net', 'ynet': 'QA-Alg'}
    
    # top of table
    table += '\n' + r'& {Name} & {Location}'
    for mn in model_names:
        name = mn if mn not in latex_name else latex_name[mn]
        table += r'& {' + name + r'}'
    table += r'\\' + '\n' + r'\midrule'

    # fill in body of table
    for k, (qr, stat_dict) in enumerate(meta_result.items()):
        is_last_row = k == len(meta_result) - 1
        
        table += '\n' + r'\multirow{' + f'{len(loc_names) * len(stat_names)}' + '}{*}{' + f'{qr[0]}-{qr[1]}' + r'}'

        for i, (sn, model_dict) in enumerate(stat_dict.items()):
            table += r'& \multirow{' + f'{len(loc_names)}' + '}{*}{' + f'{sn}' + r'}' + '\n'
            for n, (loc, model_stat) in enumerate(model_dict.items()):
                table += r' & {'+f'{loc}'+r'}'
                for i, (mean, std) in enumerate(model_stat.values()):
                    table += f' & {np.round(mean, 1)}' + r'$\pm$' + f'{np.round(std, 1)}'  
                if n != len(model_dict) - 1:
                    table += r'\\ &'

            if i != len(stat_dict) - 1:
                table = table[0:-2]
                table += '\n' + r'\cline{3-' + f'{len(model_names) + 3}' + r'}' + '\n'
            
        if not is_last_row:
            table +=  r'\cline{2-' + f'{len(model_names) + 3}' + r'}'

    # bottom of table
    table += '\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'+r'\end{table}'+'\n'
    table += r'\end{document}'

    return table  

        
def plot_epoch_history(
                            run_results, 
                            num_repeats,
                            to_plot = ['loss', 'mape', 'mse'],
                            directory='',
                            color_list=['#67597A', '#85BAA1', 'red']
                      ):    
        
    model_names = list(run_results.keys())
    
    longest_run = {mn: None for mn in model_names}
    
    # average results of repeated experiments 
    for k, (model_name, run_list) in enumerate(run_results.items()):
        fig_all, axs_all = plt.subplots(2, 1)
        
        # stat dict corresponding to longest training process 
        run_lengths = [len(stat_dict['loss']) for stat_dict in run_list]
        longest_run[model_name] = run_list[np.argmax(run_lengths)]
    
    # populate figures 
    convert_stat_names = {
                                'loss': r'$\mathbf{Loss}$', 
                                'val_loss': r'$\mathbf{Validation}$ $\mathbf{Loss}$', 
                                'mse': r'$\mathbf{MSE}$',
                                'val_mse': r'$\mathbf{Validation}$ ' + r'$\mathbf{MSE}$',
                                'mape': r'$\mathbf{MAPE}$',
                                'val_mape': r'$\mathbf{Validation}$ ' + r'$\mathbf{MAPE}$'
                         }

    figures = [plt.subplots(2, 1) for _ in range(len(to_plot))]
    max_epoch = 0 
    for n, model_name in enumerate(model_names):
        #color = color_list[n % len(color_list)]
        epoch_len = len(longest_run[model_name]['val_loss'])
        max_epoch = max(epoch_len, max_epoch)
        for (fig, axs), stat_name in zip(figures, to_plot):
            if stat_name not in convert_stat_names or stat_name not in longest_run[model_name]:
                continue 

            bottom_sn = f'val_{stat_name}'
            
            axs[0].plot(longest_run[model_name][stat_name], label=f'L{model_name.capitalize()}')
            axs[1].plot(longest_run[model_name][bottom_sn])
            
            # change to nice layout 
            stat_name = convert_stat_names[stat_name] if stat_name in convert_stat_names.keys() else stat_name
            bottom_sn = convert_stat_names[bottom_sn] if bottom_sn in convert_stat_names.keys() else bottom_sn
            
            axs[0].set_title(stat_name)
            
            axs[0].set_ylabel(r'$\mathbf{Training}$', fontsize=12)
            axs[1].set_ylabel(r'$\mathbf{Validation}$', fontsize=12)
    
    # save all figures 
    xticks = np.arange(0, max_epoch, 1)
    colors = []
    for k, (fig, axs) in enumerate(figures):
        if k == 0:
            fig.suptitle(r'$\mathbf{Training}$' + ' ' + r'$\mathbf{Statistics}$', fontsize=14)
            
        axs[0].xaxis.set_visible(False)
        axs[1].set_xlabel(r'$\mathbf{Epoch}$', fontsize=11)
        axs[0].legend(edgecolor='black', fancybox=False)
        
        fig.align_ylabels(axs)
        fig.savefig(f'{directory}/Epoch_hist_{k}.pdf')
        plt.close(fig)

            
# plot time series of a specific location 
def plot_time_series(
                         models, 
                         ximages,
                         measurements,
                         coords,
                         ntime=8,
                         lat=0,
                         lon=0,
                         name='time_series'
                    ):
    
    rainfall = measurements.isel(time=slice(0, ntime)).sel(lat=lat, lon=lon, method='nearest')
    fig, axs = plt.subplots() #[plt.subplots() for _ in range(len(models.keys()) + 1)]
    fig.suptitle(name.capitalize())
    
    inputs = ximages.isel(time=slice(0, ntime))
    
    for n, (model_name, _model) in enumerate(models.items()):
        model_pred = evaluate_models(_model, inputs, coords, lat, lon)
        axs.plot(model_pred, label=model_name)
    
    axs.plot(rainfall, label='real', color='black')
    axs.legend()    
    fig.savefig(f'')
    plt.close(fig)