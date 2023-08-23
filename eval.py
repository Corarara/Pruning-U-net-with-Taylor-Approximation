import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np
#from dice_loss import dice_coeff
from sklearn.metrics import r2_score as R2
from sklearn.metrics import mean_squared_error as Mse
#import dice_loss
from utils_p import batch
from LoadTrainTest import totorch
import csv  
import matplotlib.pyplot as plt
import colorcet as cc


def eval_net(net, dataset_inp, dataset_out, lendata, gpu=False, batch_size=1, is_loss=True):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    r2 = np.zeros(lendata)
    mse = np.zeros(lendata)

    inds = slice(278, 600)

    index_perm = np.arange(lendata)
    
    net_cpu = net.to('cpu')
    out_mod = net_cpu(dataset_inp.to('cpu')).detach().cpu().numpy()
    for index in index_perm:
        
        true_masks = dataset_out[index, :,inds, :]

        masks_pred = out_mod[index, :, inds, :]
                
        loss_r2 = R2(true_masks.flatten(), masks_pred.flatten())
        r2[index] = loss_r2
        
        loss_mse = Mse(true_masks.flatten(), masks_pred.flatten())
        mse[index] = loss_mse
        
    
    out_list = [r2, mse]
    with open('R2Test_Nbase32_final_loss.csv', 'w') as f:
        file_writer = csv.writer(f)
        file_writer.writerow(['r2', 'mse'])
        for i in range(len(r2)):
            file_writer.writerow([x[i] for x in out_list])
    
    plt.rcParams.update({'axes.titlesize': 'xx-large'})
    fig, axs = plt.subplots(1,5, figsize = (21,10))
    fig.set_dpi(512)
    
    Nind =149
    #Nind = 74
    ax = axs.ravel()
    vlim = 0.065 #Can adjust this if they saturate
    
    ssh = dataset_inp[Nind, 0, :, :]
    ssh_cos_mod,  ssh_cos_data = out_mod[Nind, 0, :, :], dataset_out[Nind, 0, :, :]
    ssh_sin_mod,  ssh_sin_data = out_mod[Nind, 1, :, :], dataset_out[Nind, 1, :, :]
    
    irectan=slice(104, 620)
    ax[0].imshow(np.flipud(ssh.cpu().numpy()[irectan,:]), vmin = -vlim*20, vmax = vlim*20, cmap =
                 cc.cm.CET_D2)
    ax[0].set_title(r'$\eta$')
    
    ax[1].imshow(np.flipud(ssh_cos_mod[irectan,:]), vmin = -vlim, vmax = vlim, cmap = cc.cm.CET_D1A)
    ax[1].set_title(r'$\eta^D_{cos}$')

    ax[2].imshow(np.flipud(ssh_cos_data[irectan,:]), vmin = -vlim, vmax = vlim, cmap = cc.cm.CET_D1A)
    ax[2].set_title(r'$\eta^{\theta}_{cos}$')

    ax[3].imshow(np.flipud(ssh_sin_mod[irectan,:]), vmin = -vlim, vmax = vlim, cmap = cc.cm.CET_D1A)
    ax[3].set_title(r'$\eta^D_{sin}$')
    
    ax[4].imshow(np.flipud(ssh_sin_data[irectan,:]), vmin = -vlim, vmax = vlim, cmap = cc.cm.CET_D1A)
    ax[4].set_title(r'$\eta^{\theta}_{sin}$')
    
    for axi in ax:
        axi.set_axis_off()
    
    fig.savefig("ssh_Nbase32_best")
    
    ###
    #index=1
    ###
    v1 = sum(r2)/index
    v2 = sum(r2[10:])/len(r2[10:])
    v3 = sum(mse)/index
    v4 = sum(mse[10:])/len(mse[10:])
    
    value = [v1,v2,v3,v4]
                
    return value, r2