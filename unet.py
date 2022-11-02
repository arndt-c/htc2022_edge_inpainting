import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning
import torch.nn.functional as F
from torchvision.utils import make_grid

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, channels, skip_channels, kernel_size = 7,
                 use_sigmoid=True, use_norm=True, num_groups=8, normalize_input=False):
        super(UNet, self).__init__()
        assert (len(channels) == len(skip_channels))
        self.scales = len(channels)
        self.use_sigmoid = use_sigmoid
        self.use_norm = use_norm
        self.num_groups = num_groups
        self.normalize_input = normalize_input
        if self.normalize_input:
            self.instance_norm = torch.nn.InstanceNorm2d(1)
        if not isinstance(kernel_size, tuple):
            self.kernel_size = [kernel_size]*self.scales
        else:
            self.kernel_size = kernel_size
        assert (len(channels)) == len(self.kernel_size)
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.inc = InBlock(in_ch, channels[0], use_norm=use_norm, kernel_size=1, num_groups=self.num_groups)
        for i in range(1, self.scales):
            self.down.append(DownBlock(in_ch=channels[i - 1],
                                       out_ch=channels[i],
                                       use_norm=use_norm,
                                       kernel_size=self.kernel_size[i],
                                       num_groups=self.num_groups))
        for i in range(1, self.scales):
            self.up.append(UpBlock(in_ch=channels[-i],
                                   out_ch=channels[-i - 1],
                                   skip_ch=skip_channels[-i],
                                   use_norm=use_norm,
                                   kernel_size=self.kernel_size[-i],
                                   num_groups=self.num_groups))
        self.outc = OutBlock(in_ch=channels[0],
                             out_ch=out_ch)
    def forward(self, x0):
        if self.normalize_input:
            x0 = self.instance_norm(x0)
        xs = [self.inc(x0), ]
        for i in range(self.scales - 1):
            xs.append(self.down[i](xs[-1]))
        x = xs[-1]
        for i in range(self.scales - 1):
            x = self.up[i](x, xs[-2 - i])
        return torch.sigmoid(self.outc(x)) if self.use_sigmoid else self.outc(x)
        
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, use_norm=True,num_groups=1):
        super(DownBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        if use_norm:
            self.conv = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_groups, num_channels=out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_groups, num_channels=out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size, # one layer more
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_groups, num_channels=out_ch),
                nn.LeakyReLU(0.2, inplace=True),)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=2, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size, # one layer more
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),)
    def forward(self, x):
        x = self.conv(x)
        return x
        
class InBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, use_norm=True, num_groups=1):
        super(InBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        if use_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x
        
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=4, kernel_size=3, use_norm=True,num_groups=1):
        super(UpBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        self.skip = skip_ch > 0
        if use_norm:
            self.conv = nn.Sequential(
                nn.GroupNorm(num_groups, num_channels=in_ch + skip_ch),
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1,
                          padding=to_pad),
                nn.GroupNorm(num_groups, num_channels=out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_groups, num_channels=out_ch),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1,
                          padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True))
        if self.skip:
            if use_norm:
                self.skip_conv = nn.Sequential(
                    nn.Conv2d(out_ch, skip_ch, kernel_size=1, stride=1),
                    nn.GroupNorm(num_groups, num_channels=skip_ch),
                    nn.LeakyReLU(0.2, inplace=True))
            else:
                self.skip_conv = nn.Sequential(
                    nn.Conv2d(out_ch, skip_ch, kernel_size=1, stride=1),
                    nn.LeakyReLU(0.2, inplace=True))
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat = Concat()
        
    def forward(self, x1, x2):
        x = self.up(x1)
        if self.skip:
            x2 = self.skip_conv(x2)
            x = self.concat(x, x2)
        x = self.conv(x)
        return x
        
class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()
    def forward(self, *inputs):
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]
       # print("CONCAT: ")
       # print("Shapes: ")
       # print("\t From down: ", inputs[0].shape)
       # print("\t From skip: ", inputs[1].shape)
        if (np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and
                np.all(np.array(inputs_shapes3) == min(inputs_shapes3))):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)
            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2,
                                   diff3:diff3 + target_shape3])
        return torch.cat(inputs_, dim=1)
        
class OutBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
    def forward(self, x):
        x = self.conv(x)
        return x
    def __len__(self):
        return len(self._modules)
        

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, use_norm=True,num_groups=1):
        super(DownBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        if use_norm:
            self.conv = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_groups, num_channels=out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_groups, num_channels=out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size, # one layer more
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_groups, num_channels=out_ch),
                nn.LeakyReLU(0.2, inplace=True),)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=2, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size, # one layer more
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),)
                
    def forward(self, x):
        x = self.conv(x)
        return x


class InBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, use_norm=True, num_groups=1):
        super(InBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        if use_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True))
                
    def forward(self, x):
        x = self.conv(x)
        return x
        
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=4, kernel_size=3, use_norm=True,num_groups=1):
        super(UpBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        self.skip = skip_ch > 0
        
        if use_norm:
            self.conv = nn.Sequential(
                nn.GroupNorm(num_groups, num_channels=in_ch + skip_ch),
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1,
                          padding=to_pad),
                nn.GroupNorm(num_groups, num_channels=out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.GroupNorm(num_groups, num_channels=out_ch),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1,
                          padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True))
        if self.skip:
            if use_norm:
                self.skip_conv = nn.Sequential(
                    nn.Conv2d(out_ch, skip_ch, kernel_size=1, stride=1),
                    nn.GroupNorm(num_groups, num_channels=skip_ch),
                    nn.LeakyReLU(0.2, inplace=True))
            else:
                self.skip_conv = nn.Sequential(
                    nn.Conv2d(out_ch, skip_ch, kernel_size=1, stride=1),
                    nn.LeakyReLU(0.2, inplace=True))
                    
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat = Concat()
        
    def forward(self, x1, x2):
        x = self.up(x1)
        if self.skip:
            x2 = self.skip_conv(x2)
            x = self.concat(x, x2)
        x = self.conv(x)
        return x
        
class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()
        
    def forward(self, *inputs):
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]
       
        if (np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and
                np.all(np.array(inputs_shapes3) == min(inputs_shapes3))):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)
            
            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2,
                                   diff3:diff3 + target_shape3])
        return torch.cat(inputs_, dim=1)
        

class OutBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = self.conv(x)
        return x
        
    def __len__(self):
        return len(self._modules)
        


# Lightning Module

class UNet_module(pytorch_lightning.LightningModule):
    '''
    A LightningModule for supervised training of a U-Net. Training and 
    validation batches should be of the form (x,y) where x is the input and y 
    is the target
    '''
    def __init__(self, in_ch, out_ch, channels, skip_channels, kernel_size = 7, 
                 use_sigmoid=True, use_norm=True, num_groups=8, normalize_input=False, lr=0.0001, resnet=False):
        super().__init__()
        self.net = UNet(in_ch, out_ch, channels, skip_channels, kernel_size = 7, 
                 use_sigmoid=True, use_norm=True, num_groups=8, normalize_input=False)
        self.lr=lr
        self.resnet = resnet
        
        if resnet:
            if in_ch != out_ch:
                print('input and output dimension of a ResNet must coincide')
            if use_sigmoid:
                print('Are you sure, you want sigmoid activation in the residual part?')
            
    def forward(self, x):
        if self.resnet:
            return self.net.forward(x) + x
        else:
            return self.net.forward(x)
    
    
    def training_step(self, batch, batch_idx):
        
        x, y = batch

        y_hat = self.forward(x)
        
        #y_hat = torch.sigmoid(y_hat)
        
        #yedge = torch.sqrt(y[:,0,:,:]**2 + y[:,1,:,:]**2)
        #yhatedge = torch.sqrt(y_hat[:,0,:,:]**2 + y_hat[:,1,:,:]**2)
        
        num1 = y.sum(dim=[2,3])
        weight = torch.div(y.shape[2]*y.shape[3] - num1, num1).unsqueeze(2).unsqueeze(3)*y
        weight[weight==0]=1
        
        bce_loss = nn.BCEWithLogitsLoss(weight=weight)
        
        loss = bce_loss(y_hat, y)
        #loss_mse = F.mse_loss(y_hat, y)
        #loss = F.l1_loss(weight*y_hat, weight*y)
        
        self.log('loss/training', loss)
        

        return loss

    def validation_step(self, batch, batch_idx):
        
        x, y = batch

        y_hat = self.forward(x)
        
        num1 = y.sum(dim=[2,3])
        weight = torch.div(y.shape[2]*y.shape[3] - num1, num1).unsqueeze(2).unsqueeze(3)*y
        weight[weight==0]=1
        
        bce_loss = nn.BCEWithLogitsLoss(weight=weight)
        
        loss = bce_loss(y_hat, y)
        
        #loss = F.mse_loss(y_hat, y)
        #loss = F.l1_loss(weight*y_hat, weight*y)
        self.log('loss/validation', loss)
        
        y_img = torch.sigmoid(y_hat)
        
        #xedge = torch.sqrt(x[:,0:1,:,:]**2 + x[:,1:2,:,:]**2)
        
        img_grid_yhat = make_grid(y_img, normalize=True, scale_each=False, padding=8, pad_value=0.2)
        img_grid_y = make_grid(y, normalize=True, scale_each=False, padding=8, pad_value=0.2)
        img_grid_x = make_grid(x[:,0:1,:,:], normalize=True, scale_each=False, padding=8, pad_value=0.2)
        
        self.logger.experiment.add_image('target edges', img_grid_y, global_step = self.current_epoch)
        self.logger.experiment.add_image('network output', img_grid_yhat, global_step = self.current_epoch)
        self.logger.experiment.add_image('network input', img_grid_x, global_step = self.current_epoch)
        
        return loss
          
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer