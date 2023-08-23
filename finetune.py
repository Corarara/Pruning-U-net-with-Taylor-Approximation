import torch
import numpy as np

from tqdm import tqdm
from utils_p import batch, AverageMeter, get_imgs_and_masks
#from flops_counter import flops_count
from flops_counter import flops_count
from LoadTrainTest import loadtrain, loadtest, totorch
import utils
import time

def finetune(net, optimizer, criterion, log, path, rand_ints, rand_start, nctrains, sizes, iters=100, epochs=None, batch_size=2, gpu=True, scale=0.5):
    net.train()
    bce_meter = AverageMeter()

    if epochs is None:  # Fine-tune using iterations of mini-batches
        epochs = 1
    else:  # Fine-tune using entire epochs
        iters = None
    
    Ntrain = 600
    Tcycle = 10
    lr0 = 0.005
    training_time = np.zeros(epochs)
    for e in tqdm(range(epochs)):
        epoch_start = time.time()
        lr = utils.cosineSGDR(optimizer, e, T0=Tcycle, eta_min=0, eta_max=lr0, scheme = 'constant')
        index_perm = np.arange(0, Ntrain, batch_size)
        for index in index_perm:
            train, true_masks = loadtrain(index, batch_size,rand_ints, rand_start, nctrains,sizes)
            train, true_masks = totorch(train), totorch(true_masks)
                
            masks_pred = net(train)
            loss = criterion(masks_pred.squeeze(),true_masks.squeeze())
            ###############
            bce_meter.update(loss.item(), batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #progress_bar.update(batch_size)
            #progress_bar.set_postfix(epoch=e, R2=bce_meter.avg)

            if index == 0 and e == 0:
                log.info("FLOPs after pruning: \n{}".format(flops_count(net, train.shape[2:])))

            if index == iters:  # Stop finetuning after sufficient mini-batches
                break
        epoch_end = time.time()
        training_time[e] = epoch_end - epoch_start
    log.info("Training time for each epoch: {}".format(training_time))
    log.info("Finished finetuning")
    log.info("Finetuned loss: {}".format(bce_meter.avg))
    torch.save(net.state_dict(), path)
    log.info('Saving finetuned to {}...'.format(path))