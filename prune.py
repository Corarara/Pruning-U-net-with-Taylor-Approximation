import torch
import torch.nn as nn
from torch import optim
import numpy as np
from netCDF4 import Dataset
import os
import os.path as osp
import json
from optparse import OptionParser
from prune_utils import Pruner
from tqdm import tqdm
from finetune import finetune
from eval import eval_net
from unet_Copy import UNet
from utils_p import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, get_logger, get_save_dir
from LoadTrainTest import loadtrain, loadtest, totorch

def get_args():
    parser = OptionParser()
    parser.add_option('-n', '--name', dest='name',
                      default="initial", help='run name')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=3, 
                      type='int', help='batch size') 
    parser.add_option('-t', '--taylor_batches', dest='taylor_batches', default=200,
                      type='int', help='number of mini-batches used to calculate Taylor criterion') #t batch?
    parser.add_option('-p', '--prune_channels', dest='prune_channels', default=150,
                      type='int', help='number of channels to remove')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda') 
    parser.add_option('-l', '--load', dest='load',
                      default="MODEL.pth", help='load file model') 
    parser.add_option('-c', '--channel_txt', dest='channel_txt',
                      default="AModel_channels.txt", help='load channel txt')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('-r', '--lr', dest='lr', type='float',
                      default=0.005, help='learning rate for finetuning') 
    parser.add_option('-i', '--iters', dest='iters', type='int',
                      default=1500, help='number of mini-batches for fine-tuning') 
    parser.add_option('-e', '--epochs', dest='epochs', type='int',
                      default=300, help='number of epochs for final finetuning')
    parser.add_option('-f', '--flops', dest='flops_reg', type='float',
                      default=.001, help='FLOPS regularization strength')
    (options, args) = parser.parse_args()
    return options


######################################################################################
if __name__ == '__main__':
    print("hhhhhhhhhhhhhhhhhh")
    
    root_dir = "data/"
    files_train = ['wp50.nc', 'wp60.nc', 'wp75.nc', 'wp80.nc']
    file_test = 'wp90.nc'
    nctrains = [Dataset(root_dir + f, 'r') for f in files_train]
    nctest = Dataset(root_dir + file_test, 'r')

    sizes = [720]
    Nchunk = len(sizes)
    rand_ints = np.random.randint(0, Nchunk, (1000,1))
    rand_start = np.random.randint(0, 200, (1000,1))
    
    # Book-keeping & paths
    args = get_args()

    dir_checkpoint = 'save/'

    runname = args.name
    save_path = os.path.join(dir_checkpoint, runname)
    save_dir = get_save_dir(save_path, runname, training=False)  # unique save dir
    log = get_logger(save_dir, runname)  # logger
    log.info('Args: {}'.format(json.dumps({"batch_size": args.batch_size,
                                           "taylor_batches": args.taylor_batches,
                                           "prune_channels": args.prune_channels,
                                           "gpu": args.gpu,
                                           "load": args.load,
                                           "channel_txt": args.channel_txt,
                                           "scale": args.scale,
                                           "lr": args.lr,
                                           "iters": args.iters,
                                           "epochs": args.epochs,
                                           "flops_reg": args.flops_reg},
                                          indent=4, sort_keys=True)))

    

    # Model Initialization
    net = UNet(n_channels=1, n_classes=2,f_c="AModel_channels.txt", bilinear = True, Nbase=16)
    log.info("Built model using {}...".format("AModel_channels.txt"))
    if args.gpu:
        net.cuda()
    if args.load:
        net.load_state_dict(torch.load(args.load))
        log.info('Loading checkpoint from {}...'.format(args.load))

    pruner = Pruner(net, args.flops_reg)  # Pruning handler

    criterion = nn.L1Loss()
          
    # Ranking on the train dataset
    log.info("Evaluating Taylor criterion for %i mini-batches" % args.taylor_batches)
    with tqdm(total=args.taylor_batches*args.batch_size) as progress_bar:
        
        Ntrain = 600
        index_perm = np.arange(0, Ntrain, args.batch_size)
        for index in index_perm:
            train, true_masks = loadtrain(index, args.batch_size, rand_ints=rand_ints,
                                          rand_start=rand_start, nctrains=nctrains,
                                          sizes=sizes)
            train, true_masks = totorch(train), totorch(true_masks)
            
            net.zero_grad()  # Zero gradients. DO NOT ACCUMULATE

            masks_pred = net(train)
            loss = criterion(masks_pred.squeeze(),true_masks.squeeze())
            loss.backward()
            
            # Compute Taylor rank
            if index==0:
                log.info("FLOPs before pruning: \n{}".format(pruner.calc_flops()))
            pruner.compute_rank()

            # Tracking progress
            progress_bar.update(args.batch_size)
            if index == args.taylor_batches:  # Stop evaluating after sufficient mini-batches
                log.info("Finished computing Taylor criterion")
                break
                
    
    # Prune & save
    pruner.pruning(args.prune_channels)
    log.info('Completed Pruning of %i channels' % args.prune_channels)

    save_file = osp.join(save_dir, "Pruned.pth")
    torch.save(net.state_dict(), save_file)
    log.info('Saving pruned to {}...'.format(save_file))

    save_txt = osp.join(save_dir, "pruned_channels.txt")
    pruner.channel_save(save_txt)
    log.info('Pruned channels to {}...'.format(save_txt))

    del net, pruner
    net = UNet(n_channels=1, n_classes=2,f_c=save_txt, bilinear = True, Nbase=16)
    log.info("Re-Built model using {}...".format(save_txt))
    if args.gpu:
        net.cuda()
    if args.load:
        net.load_state_dict(torch.load(save_file))
        log.info('Re-Loaded checkpoint from {}...'.format(save_file))
    
    optimizer=torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    
    # Use epochs or iterations for fine-tuning
    save_file = osp.join(save_dir, "Finetuned.pth")
    
    
    finetune(net, optimizer, criterion, log, save_file, rand_ints, rand_start, nctrains,sizes, args.iters, args.epochs, args.batch_size, args.gpu, args.scale)
    

    #######
    ytest_slice = slice(0, 720)
    val, out_test = loadtest(nctest=nctest, ytest_slice=ytest_slice)
    val = totorch(val)
    #out_test = totorch(out_test)
    #######
    
    
    avg_r2_mse, r2 = eval_net(net, val, out_test, len(val), args.gpu, 1)
    
    log.info('Avg R2: {}'.format(avg_r2_mse))
    print(avg_r2_mse)