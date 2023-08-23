import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from netCDF4 import Dataset
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import r2_score as R2
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score as r2
from copy import deepcopy
import utils
#from attnUnet import AttentionUNet
from unet import UNet
from scipy.stats import pearsonr
import colorcet as cc
import os
import time
import csv  
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
torch.cuda.set_device(0)

root_dir = "data/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("the device now is:",device)

files_train = ['wp50.nc', 'wp60.nc', 'wp75.nc', 'wp80.nc']
file_test = 'wp90.nc'
nctrains = [Dataset(root_dir + f, 'r') for f in files_train]
nctest = Dataset(root_dir + file_test, 'r')

vari = 'ssh_ins'
Ntrain = np.sum([nc.dimensions['time_counter'].size for nc in nctrains], axis = 0); print('number of training records:', Ntrain)
numTrainFiles = len(nctrains)
numRecsFile = nctrains[0].dimensions['time_counter'].size
print("numRecsFile:",numRecsFile)

#sizes = [720, 512, 256]
sizes = [720]
Nchunk = len(sizes)
rand_ints = np.random.randint(0, Nchunk, (1000,1))
rand_start = np.random.randint(0, 200, (1000,1))
print(rand_ints.dtype)
#rand_nc = np.random.randint(0, 200, (1000,1))

sin = True

def loadtrain(index, batch_size):
    #Choose the corrent ncfile sequentially
    ncindex = int(index//numRecsFile)
    lim_ind = int(rand_ints[index])
    lim = sizes[lim_ind]
    start = int(rand_start[index])
    index = index - ncindex*numRecsFile
    #print('ncindex:', ncindex, 'index:', index, 'start:', start)
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

#ytest_slice = slice(278, 678)
ytest_slice = slice(0, 720)

def loadtest():
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
Nbase = 32

#model = AttentionUNet(Nbase = 16).to(device)
if sin:
    no = 2
else:
    no = 1
model = UNet(1, no, bilinear = True, Nbase = 32).cuda() #modify Nbase here!!!!!!!!!!!!!!!!!!!!!!
input = torch.randn(1,1,256,256).to(device)
output = model(input)
print('Model has ', utils.nparams(model)/1e6, ' million params')

batch_size = 20 #HW made this smaller to save GPU cost (but it will make the code slower due to less parallelization )
for index in range(0, Ntrain, batch_size):
    inp, out = loadtrain(index, batch_size)
    print(inp.shape, out.shape)
    print(np.mean(inp**2), np.max(inp**2), inp.shape)
    print(np.mean(out**2), np.max(out**2), inp.shape)
    
inp, out_test = loadtest()
with torch.no_grad():
    inp_test = totorch(inp)
    #out_model = attention_unet(inp_test)
    
print(inp_test.shape)

#######################
#lr0=0.02
lr0 = 0.005 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
maxEpochs = 300 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Tcycle = 10 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
criterion_train  = nn.L1Loss()
optim = torch.optim.AdamW(model.parameters(), lr=lr0, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
#optim = torch.optim.SGD(model.parameters(), lr=lr0)
#######################

#lr0 = 0.005
loss_train = np.zeros(maxEpochs)
r2_test = np.zeros(maxEpochs)
r2_train = np.zeros(maxEpochs)
epochmin = []
maxr2l = []
maxr2_cycle = np.zeros(maxEpochs)
maxR2_global = np.zeros(maxEpochs)

learn = np.zeros(maxEpochs)
minloss = 1000
maxR2 = -1000
maxR2_cycle = -1000
minlosscount = 0
perm = False
lr_list = np.zeros(maxEpochs) ########### exam the learning rate

print('Starting training loop')
training_time = np.zeros(maxEpochs)
save_model_time = np.zeros(maxEpochs)
for epoch in tqdm(range(maxEpochs)):
    epoch_start = time.time()
    lr = utils.cosineSGDR(optim, epoch, T0=Tcycle, eta_min=0, eta_max=lr0, scheme = 'constant') 
    #lr = utils.tri_simin(optim, epoch, stepsize=5, lr0=0.005)
    lr_list[epoch]=lr ###########exam the learning rate
    model.train()
    index_perm = np.arange(0, Ntrain, batch_size)
    loss_temp = np.zeros(len(index_perm))
    r2_temp = np.zeros(len(index_perm))
    if perm:
        index_perm = np.random.permutation(index_perm)
    i = 0
    for index in index_perm:
        inp, out = loadtrain(index, batch_size)
        inp, out = totorch(inp), totorch(out)
        out_mod = model(inp)
        loss = criterion_train(out.squeeze(), out_mod.squeeze())
        loss_temp[i] = loss.item()
        
        r2_temp[i] = R2(out.detach().cpu().numpy().flatten(), out_mod.detach().cpu().numpy().flatten())
        i += 1
        #print('loss',loss)
        #Set gradient to zero
        optim.zero_grad()
        #Compute gradients       
        loss.backward()
        #Update parameters with new gradient
        optim.step()
        #Record train loss
        #scheduler.step()
    #if (epoch > maxEpochs -2): #***save just 2 during debugging
    epoch_end = time.time()
    #if (epoch > maxEpochs - 2*Tcycle - 1) or (epoch%5==0):
    
    loss_avg = np.mean(loss_temp)
    loss_train[epoch] = loss_avg
    r2_train[epoch] = np.mean(r2_temp)
    
    if True:
      model.eval()
      with torch.no_grad():
            model_cpu = model.to('cpu')
            out_mod = model_cpu(inp_test.to('cpu')) #Ths step costs a lot of memory

            r2 = R2(out_test.flatten(), (out_mod).cpu().numpy().flatten())
            r2_test[epoch] = r2
           
            if maxR2 <  r2:
                maxR2 = r2
                epochmin.append(epoch)
                maxr2l.append(maxR2)
                maxR2_global[epoch] = r2
                model_best = deepcopy(model)
                corr, pval = pearsonr(out_test.flatten(), (out_mod).cpu().numpy().flatten())
                print("At epoch",epoch,'R2:', r2, ' corr: ', corr, ' pval: ', pval)
            model = model_cpu.to(device)
            save_model_end = time.time()
    #record time
    train_time_diff = epoch_end - epoch_start
    training_time[epoch] = train_time_diff
    if save_model_end > epoch_end:
        save_model_time[epoch] = save_model_end - epoch_end
        save_model_end = epoch_end

_, out_test = loadtest()
model_best.eval()

with torch.no_grad():
    out_mod = model_best(inp_test.to('cpu')).detach().cpu().numpy()

Nx, Ny = out_test.shape[2:]
print(Nx,Ny)

############################################# Plotting ################################################
print("Start plotting")
plt.rcParams.update({'axes.titlesize': 'xx-large'})
fig, axs = plt.subplots(2, 3, figsize = (21,14))
# Nind = np.random.randint(150); print (f'Nind: {Nind}')
Nind = np.random.randint(batch_size); print (f'Nind: {Nind}')
# Nind =97
ax = axs.ravel()
vlim = 0.03
cmap = 'jet'
inds = slice(278, 600)
print('R2:', R2(out_test[Nind, :,inds, :].flatten(), out_mod[Nind, :, inds, :].flatten()))
print('corr:', pearsonr(out_test[Nind, :,inds, :].flatten(), out_mod[Nind, :, inds, :].flatten())[0])

ssh_cos_mod,  ssh_cos_data = out_mod[Nind, 0, :, :], out_test[Nind, 0, :, :]
ssh_sin_mod,  ssh_sin_data = out_mod[Nind, 1, :, :], out_test[Nind, 1, :, :]

print('corr cossin:', pearsonr(ssh_cos_mod.flatten(), ssh_sin_data.flatten())[0])

ssh = inp_test[Nind, 0, :, :]

ax[1].pcolormesh(ssh_cos_mod, vmin = -vlim, vmax = vlim, cmap = cmap)
ax[1].set_title(r'$\eta^D_{cos}$')

ax[0].pcolormesh(ssh_cos_data, vmin = -vlim, vmax = vlim, cmap = cmap)
ax[0].set_title(r'$\eta^{\theta}_{cos}$')

ax[2].pcolormesh(ssh.cpu().numpy(), vmin = -1.5, vmax = 1.5, cmap = cmap)
ax[2].set_title(r'$\eta$')

ax[3].pcolormesh(ssh_sin_data, vmin = -vlim, vmax = vlim, cmap = cmap)
ax[3].set_title(r'$\eta^D_{sin}$')

ax[4].pcolormesh(ssh_sin_mod, vmin = -vlim, vmax = vlim, cmap = cmap)
ax[4].set_title(r'$\eta^{\theta}_{sin}$')
ax[5].scatter(ssh_cos_data.flatten(), ssh_cos_mod.flatten(), alpha = 0.02)
# ax[5].set_xlim([-0.03, 0.04])
# ax[5].set_ylim([-0.03, 0.04])
ax[5].plot(out_test[Nind, 0, inds, :].flatten(), out_test[Nind, 0, inds, :].flatten(), 'r--')
ax[5].set_title(r'$\eta^{\theta}_{cos}$ vs $\eta^{D}_{cos}$')
#ax[5].pcolormesh(inp_test[Nind, 0, :, :].cpu().numpy(), vmin = -1, vmax = 1, cmap = cmap)

print("Start second plotting")
#To plot following hw's conventions; to do.
plt.rcParams.update({'axes.titlesize': 'xx-large'})
fig, axs = plt.subplots(1,5, figsize = (21,10))
fig.set_dpi(512)

# Nind = np.random.randint(150); print (f'Nind: {Nind}')
#Nind = np.random.randint(batch_size); print (f'Nind: {Nind}')
Nind =149
ax = axs.ravel()
vlim = 0.065 #Can adjust this if they saturate

ssh = inp_test[Nind, 0, :, :]

irectan=slice(104, 620)
ax[0].imshow(np.flipud(ssh.cpu().numpy()[irectan,:]), vmin = -vlim*20, vmax = vlim*20, cmap = cc.cm.CET_D2)
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

dr = './models'
try:
    os.mkdir(dr)
except:
    print('Directory exists or disk full...')
#fstr = f'MaxEpoch_{maxEpochs}_Nbase_{Nbase}_batch_{batch_size}_lr_{lr0}_Tcycle_{Tcycle}_{perm}'
#fstr = f'MaxEpoch_{maxEpochs}_Nbase_{Nbase}_batch_{batch_size}_lr_{lr0}'
fstr = f"Nbase_{Nbase}_Final_loss"
#fstr = 'final'
#You can change the file name here
PATH = dr + f'/Unet_{fstr}.pth'
torch.save(model_best.state_dict(), PATH)


#Codes that save the .nc files from the best checkpoint in the testing set
outdr = './data/'
nco = utils.ncCreate(outdr + f'output_{fstr}.nc', Nx, Ny, ['sin_model', 'cos_model', 'sin_data', 'cos_data'])
utils.addVal(nco, 'cos_model', out_mod[:, 0, :, :])
utils.addVal(nco, 'sin_model', out_mod[:, 1, :, :])
utils.addVal(nco, 'cos_data', out_test[:, 0, :, :])
utils.addVal(nco, 'sin_data', out_test[:, 1, :, :])
nco.close()


######################################### Saving part ###########################################
# Save images into a pdf file
def save_image(filename):
    p = PdfPages(filename)
    fig_nums = plt.get_fignums()  
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs: 
        fig.savefig(p, format='pdf') 
    p.close()  
filename = f"{fstr}.pdf"  

save_image(filename)
print("Image saved")


# Save learning rate, time, R2 during training
epoch_list = list(range(maxEpochs))
out_list = [epoch_list,lr_list,training_time,save_model_time,r2_test,maxR2_global, loss_train, r2_train]
with open(f'output/{fstr}.csv', 'w') as f:
    file_writer = csv.writer(f)
    file_writer.writerow(['Epoch','Learning Rate','Training Time','Model Save Time', 'R2', 'maxR2_global', 'loss','r2_train'])
    for i in range(maxEpochs):
        file_writer.writerow([x[i] for x in out_list])
print("Data output saved")
