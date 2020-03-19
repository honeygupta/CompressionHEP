'''
Modified by Honey Gupta. The original scripts can be found at https://github.com/erwulff/lth_thesis_project and https://github.com/Skelpdar/HEPAutoencoders.

Functions were modified or added for 4D data. The ones related to 27D AOD data were removed for better readibility.
'''

import time
import pickle
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import TensorDataset

import my_matplotlib_style as ms

from fastai import basic_data, basic_train
from fastai import train as tr

# Functions for evaluation
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rms(arr):
    arr = arr.flatten()
    arr[arr == np.nan] = 1
    return np.sqrt(np.sum(arr**2) / len(arr))


def nanrms(x, axis=None):
    return np.sqrt(np.nanmean(x**2, axis=axis))


def std_error(x, axis=None, ddof=0):
    return np.nanstd(x, axis=axis, ddof=ddof) / np.sqrt(2 * len(x))


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def validate(model, dl, loss_func):
    for batch in dl:
        losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(val_loss)
        return val_loss

# Plotting functions
def plot_residuals(pred, data, range=None, variable_names=['pT', 'eta', 'phi', 'E'], bins=1000, save=None, title=None):
    alph = 0.8
    residuals = (pred.numpy() - data.numpy()) / data.numpy()
    for kk in np.arange(4):
        plt.figure()
        n_hist_pred, bin_edges, _ = plt.hist(residuals[:, kk], label='Residuals', alpha=alph, bins=bins, range=range)
        if title is None:
            plt.suptitle('Residuals of %s' % variable_names[kk])
        else:
            plt.suptitle(title)
        plt.xlabel(r'$(%s_{recon} - %s_{true}) / %s_{true}$' % (variable_names[kk], variable_names[kk], variable_names[kk]))
        plt.ylabel('Number of events')
        ms.sciy()
        if save is not None:
            plt.savefig(save + '_%s' % variable_names[kk])


def plot_histograms(pred, data, bins, same_bin_edges=True, colors=['orange', 'c'], variable_list=[r'$p_T$', r'$\eta$', r'$\phi$', r'$E$'], variable_names=['pT', 'eta', 'phi', 'E'], unit_list=['[GeV]', '[rad]', '[rad]', '[GeV]'], title=None):
    alph = 0.8
    n_bins = bins
    for kk in np.arange(4):
        plt.figure()
        n_hist_data, bin_edges, _ = plt.hist(data[:, kk], color=colors[1], label='Input', alpha=1, bins=n_bins)
        if same_bin_edges:
            n_bins_2 = bin_edges
        else:
            n_bins_2 = bins
        n_hist_pred, _, _ = plt.hist(pred[:, kk], color=colors[0], label='Output', alpha=alph, bins=n_bins_2)
        if title is None:
            plt.suptitle(variable_names[kk])
        else:
            plt.suptitle(title)
        plt.xlabel(variable_list[kk] + ' ' + unit_list[kk])
        plt.ylabel('Number of events')
        ms.sciy()
        plt.legend()


def plot_activations(learn, figsize=(12, 9), lines=['-', ':'], save=None, linewd=1, fontsz=14):
    plt.figure(figsize=figsize)
    for i in range(learn.activation_stats.stats.shape[1]):
        thiscol = ms.colorprog(i, learn.activation_stats.stats.shape[1])
        plt.plot(learn.activation_stats.stats[0][i], linewidth=linewd, color=thiscol, label=str(learn.activation_stats.modules[i]).split(',')[0], linestyle=lines[i % len(lines)])
    plt.title('Weight means')
    plt.legend(fontsize=fontsz)
    plt.xlabel('Mini-batch')
    if save is not None:
        plt.savefig(save + '_means')
    plt.figure(figsize=(12, 9))
    for i in range(learn.activation_stats.stats.shape[1]):
        thiscol = ms.colorprog(i, learn.activation_stats.stats.shape[1])
        plt.plot(learn.activation_stats.stats[1][i], linewidth=linewd, color=thiscol, label=str(learn.activation_stats.modules[i]).split(',')[0], linestyle=lines[i % len(lines)])
    plt.title('Weight standard deviations')
    plt.xlabel('Mini-batch')
    plt.legend(fontsize=fontsz)
    if save is not None:
        plt.savefig(save + '_stds')

# Custom normalization for AOD data
m_div = 1.8
m_add = 1
pt_sub = 1.3
pt_div = 1.2
eta_div = 5
phi_div = 3

emfrac_div = 1.6
negE_div = 1.6
width_div = .6
N90_div = 20
timing_div = 40
hecq_div = 1
centerlambda_div = 2
secondlambda_div = 1
secondR_div = .6
larqf_div = 2.5
centroidR_div = 0.8
area4vecm_div = 0.18
area4vecpt_div = 0.7
area4vec_div = 0.8
Oot_div = 0.3
larq_div = 0.6
log_add = 100
log_sub = 2
centroidR_sub = 3
area4vecm_sub = 0.15

#     <--- Modified by Honey for 4D data --->
def custom_normalization(train, test):
    train_cp = train.copy()
    test_cp = test.copy()
    
    for data in [train_cp, test_cp]:
        data['pt'] = (np.log10(data['pt']) - pt_sub) / pt_div
        data['phi'] = data['phi'] / phi_div
        data['eta'] = data['eta'] / eta_div
        data['m'] = np.log10(data['m'] + m_add) / m_div
        
        # For the case of 27 parameter data
        if len(data.keys()) > 4:
            data['DetectorEta'] = data['DetectorEta'] / eta_div
            data['ActiveArea4vec_eta'] = data['ActiveArea4vec_eta'] / eta_div
            data['EMFrac'] = data['EMFrac'] / emfrac_div
            data['NegativeE'] = np.log10(-data['NegativeE'] + 1) / negE_div
        

            data['ActiveArea4vec_phi'] = data['ActiveArea4vec_phi'] / phi_div
            if 'Width' in data.keys():
                data['Width'] = data['Width'] / width_div
            else:
                print('Wdith not found when normalizing')
            if 'WidthPhi' in data.keys():
                data['WidthPhi'] = data['WidthPhi'] / width_div
            else:
                print('WdithPhi not found when normalizing')
            data['N90Constituents'] = data['N90Constituents'] / N90_div
            data['Timing'] = data['Timing'] / timing_div
            data['HECQuality'] = data['HECQuality'] / hecq_div
            data['ActiveArea'] = data['ActiveArea'] / area4vec_div
            data['ActiveArea4vec_m'] = data['ActiveArea4vec_m'] / area4vecm_div - area4vecm_sub
            data['ActiveArea4vec_pt'] = data['ActiveArea4vec_pt'] / area4vecpt_div
            data['LArQuality'] = data['LArQuality'] / larq_div
            data['LeadingClusterCenterLambda'] = (np.log10(data['LeadingClusterCenterLambda'] + log_add) - log_sub) / centerlambda_div
            data['LeadingClusterSecondLambda'] = (np.log10(data['LeadingClusterSecondLambda'] + log_add) - log_sub) / secondlambda_div
            data['LeadingClusterSecondR'] = (np.log10(data['LeadingClusterSecondR'] + log_add) - log_sub) / secondR_div
            data['AverageLArQF'] = (np.log10(data['AverageLArQF'] + log_add) - log_sub) / larqf_div
        
            data['LeadingClusterPt'] = np.log10(data['LeadingClusterPt']) / pt_div
            data['CentroidR'] = (np.log10(data['CentroidR']) - centroidR_sub) / centroidR_div
            data['OotFracClusters10'] = np.log10(data['OotFracClusters10'] + 1) / Oot_div
            data['OotFracClusters5'] = np.log10(data['OotFracClusters5'] + 1) / Oot_div
        
    return train_cp, test_cp

'''
<--- Added by Honey for 4D data --->
    Key:
    dim 0 = m
    dim 1 = pt
    dim 2 = phi
    dim 3 = eta
'''
def custom_unnormalize_4m(normalized_data):
    data = normalized_data.copy()
    data[:,0] = np.power(10, data[:,0]) - m_add 
    data[:,1] = np.power(10, data[:,1]) + pt_sub 
    data[:,2] = data[:,2] * phi_div
    data[:,3] = data[:,3] * eta_div
    return data
    

#     <--- Modified by Honey for 4D data --->
def custom_unnormalize(normalized_data):
    data = normalized_data.copy()

    data['pt'] = np.power(10, pt_div * data['pt'] + pt_sub)
    data['eta'] = data['eta'] * eta_div
    data['phi'] = data['phi'] * phi_div
    data['m'] =  np.power(10, m_div * data['m']) - m_add

    # For the case of 27 parameter data
    if len(data.keys()) > 4:
        data['DetectorEta'] = data['DetectorEta'] * eta_div
        data['ActiveArea4vec_eta'] = data['ActiveArea4vec_eta'] * eta_div
        data['EMFrac'] = data['EMFrac'] * emfrac_div
    
        data['ActiveArea4vec_phi'] = data['ActiveArea4vec_phi'] * phi_div
        if 'Width' in data.keys():
            data['Width'] = data['Width'] * width_div
        else:
            print('Width not found when unnormalizing')
        if 'WidthPhi' in data.keys():
            data['WidthPhi'] = data['WidthPhi'] * width_div
        else:
            print('WidthPhi not found when unnormalizing')
        data['N90Constituents'] = data['N90Constituents'] * N90_div
        data['Timing'] = data['Timing'] * timing_div
        data['HECQuality'] = data['HECQuality'] * hecq_div
        data['ActiveArea'] = data['ActiveArea'] * area4vec_div
        data['ActiveArea4vec_m'] = (data['ActiveArea4vec_m'] + area4vecm_sub) * area4vecm_div
        data['ActiveArea4vec_pt'] = data['ActiveArea4vec_pt'] * area4vecpt_div
        data['LArQuality'] = data['LArQuality'] * larq_div

        data['NegativeE'] = 1 - np.power(10, negE_div * data['NegativeE'])
    
        data['LeadingClusterCenterLambda'] = np.power(10, centerlambda_div * data['LeadingClusterCenterLambda'] + log_sub) - log_add
        data['LeadingClusterSecondLambda'] = np.power(10, secondlambda_div * data['LeadingClusterSecondLambda'] + log_sub) - log_add
        data['LeadingClusterSecondR'] = np.power(10, secondR_div * data['LeadingClusterSecondR'] + log_sub) - log_add
        data['AverageLArQF'] = np.power(10, larqf_div * data['AverageLArQF'] + log_sub) - log_add
    
        data['LeadingClusterPt'] = np.power(10, pt_div * data['LeadingClusterPt'])
        data['CentroidR'] = np.power(10, centroidR_div * data['CentroidR'] + centroidR_sub)
        data['OotFracClusters10'] = np.power(10, Oot_div * data['OotFracClusters10']) - 1
        data['OotFracClusters5'] = np.power(10, Oot_div * data['OotFracClusters5']) - 1

    return data

def round_to_input(pred, uniques, variable):
    var = pred[variable].values.reshape(-1, 1)
    diff = (var - uniques)
    ind = np.apply_along_axis(lambda x: np.argmin(np.abs(x)), axis=1, arr=diff)
    new_arr = -np.ones_like(var)
    for ii in np.arange(new_arr.shape[0]):
        new_arr[ii] = uniques[ind[ii]]
    pred[variable] = new_arr


