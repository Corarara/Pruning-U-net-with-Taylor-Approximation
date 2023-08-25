from netCDF4 import Dataset
import numpy as np
import torch
import torch.nn as nn

def loadtrain(index, batch_size,rand_ints, rand_start, nctrains,sizes):
    
    sin = True
    
    #Choose the corrent ncfile sequentially
    ncindex = int(index//150)
    lim_ind = int(rand_ints[index])
    lim = sizes[lim_ind]
    start = int(rand_start[index])
    index = index - ncindex*150

    #select a random spatial section of the flow
    rec_slice = slice(index, index + batch_size)
    yslice = slice(start, lim + start)
    if lim_ind>0:
        ssh = nctrains[ncindex].variables['ssh_ins'][rec_slice, yslice, :256]
        sshcos =  nctrains[ncindex].variables['ssh_cos'][rec_slice, yslice, :256] 
        if sin:
            sshsin =  nctrains[ncindex].variables['ssh_sin'][rec_slice, yslice, :256] 

    else:
        ssh = nctrains[ncindex].variables['ssh_ins'][rec_slice, :lim, :256]
        sshcos = nctrains[ncindex].variables['ssh_cos'][rec_slice, :lim, :256]
        if sin:
            sshsin = nctrains[ncindex].variables['ssh_sin'][rec_slice, :lim, :256]
    if sin:
        out = np.concatenate([sshcos[:, None, :, :], sshsin[:, None, :, :]], axis = 1)
        return ssh[:, None, :, :].data, out.data   

    else:
        return ssh[:, None, :, :].data, sshcos[:, None, :, :].data



def loadtest(nctest, ytest_slice):
    sin = True
    ssh = nctest.variables['ssh_ins'][:, ytest_slice, :256]
    sshcos = nctest.variables['ssh_cos'][:, ytest_slice, :256]
    if sin:
        sshsin = nctest.variables['ssh_sin'][:, ytest_slice, :256]
        out = np.concatenate([sshcos[:, None, :, :], sshsin[:, None, :, :]], axis = 1)
        return ssh[:, None, :, :].data, out.data   
    else:
        return ssh[:, None, :, :].data, sshcos[:, None, :, :].data   

    
def totorch(x):
    return torch.tensor(x, dtype = torch.float).cuda()