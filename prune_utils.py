import torch
import torch.nn as nn
import csv  


class Pruner:
    def __init__(self, net, flops_reg):
        self.net = net.eval()
        # Initialize stuff
        self.flops_reg = flops_reg
        self.clear_rank()
        self.clear_modules()
        self.clear_cache()
        # Set hooks
        self._register_hooks()

    def clear_rank(self):
        self.ranks = {}  # accumulates Taylor ranks for modules
        self.flops = []

    def clear_modules(self):
        self.convs = []
        self.BNs = []

    def clear_cache(self):
        self.activation_maps = []
        self.gradients = []

    def _register_hooks(self):
        def forward_hook_fn(module, input, output):
            """ Stores the forward pass outputs (activation maps)"""
            self.activation_maps.append(output.clone().detach())

        def backward_hook_fn(module, grad_in, grad_out):
            """Stores the gradients wrt outputs during backprop"""
            self.gradients.append(grad_out[0].clone().detach())

        for name, module in self.net.named_modules():
            if isinstance(module, nn.Conv2d):
                if name != "outc.conv":  # don't hook final conv module
                    module.register_full_backward_hook(backward_hook_fn)
                    module.register_forward_hook(forward_hook_fn)
                self.convs.append(module)
            if isinstance(module, nn.BatchNorm2d):
                self.BNs.append(module)  # save corresponding BN layer

    def compute_rank(self):  # Compute ranks after each minibatch
        self.gradients.reverse()

        for layer, act in enumerate(self.activation_maps):
            taylor = (act*self.gradients[layer]).mean(dim=(2, 3)).abs().mean(dim=0)  # C

            if layer not in self.ranks.keys():  # no such entry
                self.ranks.update({layer: taylor})
            else:
                self.ranks[layer] = .9*self.ranks[layer] + .1*taylor  # C
        self.clear_cache()

    def _rank_channels(self, prune_channels):
        total_rank = []  # flattened ranks of each channel, all layers
        channel_layers = []  # layer num for each channel
        layer_channels = []  # channel num wrt layer for each channel
        
        layer_min_rank = []
        layer_median_rank = []
        layer_max_rank = []
        
        layer_min_rank_before_norm = []
        layer_median_rank_before_norm = []
        layer_max_rank_before_norm = []
        
        self.flops[:] = [x / sum(self.flops) for x in self.flops]  # Normalize FLOPs
        
        
        for layer, ranks in self.ranks.items():
            # Average across minibatches
            taylor = ranks  # C
            
            # For plotting layers' ranks before normalization
            layer_min_rank_before_norm.append(ranks.min().item())
            layer_median_rank_before_norm.append(ranks.median().item())
            layer_max_rank_before_norm.append(ranks.max().item())
            
            # Layer-wise L2 normalization
            taylor = taylor / torch.sqrt(torch.sum(taylor**2))  # C
            
            temp = taylor-self.flops[layer]*self.flops_reg
            total_rank.append(temp) # FLOPs regularization
            
            layer_min_rank.append(temp.min().item())
            layer_median_rank.append(temp.median().item())
            layer_max_rank.append(temp.max().item())
            
            channel_layers.extend([layer]*ranks.shape[0])
            layer_channels.extend(list(range(ranks.shape[0])))
        
        channel_layers = torch.Tensor(channel_layers)
        layer_channels = torch.Tensor(layer_channels)
        total_rank = torch.cat(total_rank, dim=0)

        # Rank
        sorted_rank, sorted_indices = torch.topk(total_rank, prune_channels, largest=False)
        sorted_channel_layers = channel_layers[sorted_indices]
        sorted_layer_channels = layer_channels[sorted_indices]
        
        out_list = [layer_min_rank, layer_median_rank, layer_max_rank,
                    layer_min_rank_before_norm, layer_median_rank_before_norm, 
                    layer_max_rank_before_norm]
        
        with open('layer_rank_prune150Final.csv', 'w') as f:
            file_writer = csv.writer(f)
            file_writer.writerow(['layer_min_rank','layer_median_rank','layer_max_rank',
                                 'layer_min_rank_before_norm', 'layer_median_rank_before_norm',
                                 'layer_max_rank_before_norm'])
            for i in range(len(layer_min_rank)):
                file_writer.writerow([x[i] for x in out_list])
                
        return sorted_channel_layers, sorted_layer_channels

    def pruning(self, prune_channels):

        sorted_channel_layers, sorted_layer_channels = self._rank_channels(prune_channels)
        inchans, outchans = self.create_indices()
        #find in_channels and out_channels in conv2d
        #in_channels
        #[1,16,32,16,32,64,32,64,128,64,128,256,128,128,256,128,128,64,64,32,32,16,16]
        #out_channels
        #[16,16,16,32,32,32,64,64,64,128,128,128,128,128,128,64,64,32,32,16,16,16,2]

        
        for i in range(len(sorted_channel_layers)):
            cl = int(sorted_channel_layers[i])
            lc = int(sorted_layer_channels[i])
           
            # These tensors are concat at a later conv2d
            res = True if cl in [1, 4, 7, 10] else False

            # These tensors are concat with an earlier tensor at bottom.
            offset = True if cl in [13, 15, 17, 19] else False

            # Remove indices of pruned parameters/channels
            if offset:
                mapping = {13: 10, 15: 7, 17: 4, 19: 1}
                
                top = self.convs[mapping[cl]].weight.shape[0]
                try:
                    inchans[cl + 1].remove(top + lc)  # it is searching for a -ve number to remove, but there are none
                    # However, the output channel of the previous layer (d4) is reduced
                    # So up1's input channel is larger than expected due to failed removal
                    
                except ValueError:
                    pass
            else:
                try:
                    if cl==1: 
                        print(cl,"inchans[cl+1]",inchans[cl + 1])
                    inchans[cl + 1].remove(lc) 
                    if cl==1:
                        print(cl,"inchans[cl+1]",inchans[cl + 1])
                except ValueError:
                    pass
                
            if res:
                try:
                    #if cl==1:
                        #print(cl,"inchans[cl+1]",inchans[cl + 1])
                    top = self.convs[cl].weight.shape[0]
                    inchans[cl+1].remove(lc+top)
                    #if cl==1:
                        #print(cl,"inchans[cl+1]",inchans[cl + 1])
                except ValueError:
                    pass
                
                if cl == 1:
                    try:
                        inchans[20].remove(lc)
                    except ValueError:
                        pass
                elif cl == 4:
                    try:
                        inchans[18].remove(lc)
                    except ValueError:
                        pass
                elif cl == 7:
                    try:
                        inchans[16].remove(lc)
                    except ValueError:
                        pass
                else:
                    try:
                        inchans[14].remove(lc)
                    except ValueError:
                        pass
                    
            try:
                outchans[cl].remove(lc) #剪当前的outchan
            except ValueError:
                pass
        

        #23Conv2d: 2*4down + 2*4up + outc + 4*mixpooling
        #19BN：inpBN + 2*4down + 2*4up
        
        # Use indexing to get rid of parameters
        for i, c in enumerate(self.convs):
            self.convs[i].weight.data = c.weight[outchans[i], ...][:, inchans[i], ...]
            if i in [2,5,8,11]:
                self.convs[i].bias.data = c.bias[outchans[i]]
        
        for i, bn in enumerate(self.BNs):
            #[0:1, 1:2, 3:3, 4:4, 6:5, 7:6, 9:7, 10:8, 12:9, 13:10, 14:11, 15:12,
            # 16:13, 17:14, 18:15, 19:16, 20:17, 21:18]
     
            if i in [1,2]:
                self.BNs[i].running_mean.data = bn.running_mean[outchans[i-1]]
                self.BNs[i].running_var.data = bn.running_var[outchans[i-1]]
                self.BNs[i].weight.data = bn.weight[outchans[i-1]]
                self.BNs[i].bias.data = bn.bias[outchans[i-1]]
            elif i in [3,4]:
                self.BNs[i].running_mean.data = bn.running_mean[outchans[i]]
                self.BNs[i].running_var.data = bn.running_var[outchans[i]]
                self.BNs[i].weight.data = bn.weight[outchans[i]]
                self.BNs[i].bias.data = bn.bias[outchans[i]]
            elif i in [5,6]:
                self.BNs[i].running_mean.data = bn.running_mean[outchans[i+1]]
                self.BNs[i].running_var.data = bn.running_var[outchans[i+1]]
                self.BNs[i].weight.data = bn.weight[outchans[i+1]]
                self.BNs[i].bias.data = bn.bias[outchans[i+1]]
            elif i in [7,8]:
                self.BNs[i].running_mean.data = bn.running_mean[outchans[i+2]]
                self.BNs[i].running_var.data = bn.running_var[outchans[i+2]]
                self.BNs[i].weight.data = bn.weight[outchans[i+2]]
                self.BNs[i].bias.data = bn.bias[outchans[i+2]]
            elif i in [9,10,11,12,13,14,15,16,17,18]:
                self.BNs[i].running_mean.data = bn.running_mean[outchans[i+3]]
                self.BNs[i].running_var.data = bn.running_var[outchans[i+3]]
                self.BNs[i].weight.data = bn.weight[outchans[i+3]]
                self.BNs[i].bias.data = bn.bias[outchans[i+3]]
                
    def create_indices(self):
        chans = [(list(range(c.weight.shape[1])), list(range(c.weight.shape[0]))) for c in self.convs]
        inchans, outchans = list(zip(*chans))
        return inchans, outchans

    def channel_save(self, path):
        """save the distinct number of channels"""
        chans = []
        for i, c in enumerate(self.convs[1:-1]):
            if i in [0,1,4,7,10,13,15,17,19]:
                chans.append(c.weight.shape[1])
                
            chans.append(c.weight.shape[0])

        with open(path, 'w') as f:
            for item in chans:
                f.write("%s\n" % item)

    def calc_flops(self):
        """Calculate flops per tensor channel. Only consider flops
        of conv2d that produces said feature map
        """
        # conv2d: slides*(kernel mult + kernel sum + bias)
        # kernel_sum = kernel_mult - 1
        # conv2d: slides*(2*kernel mult)

        # batchnorm2d: 4*slides

        # Remove unnecessary constants from calculation

        for i, c in enumerate(self.convs[:-1]):
            H, W = self.gradients[i].shape[2:]
            O, I, KH, KW = c.weight.shape
            self.flops.append(H*W*KH*KW*I)
        return self.flops