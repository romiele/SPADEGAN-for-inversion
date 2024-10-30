import torch.nn as nn
import sys
import torch.nn.utils.spectral_norm as SpectralNorm
import numpy as np
import torch
import torch.nn.functional as F

def conv3x3(ch_in,ch_out,SN = False,s = 1,p=1,bias = True):
    if SN:
        return SpectralNorm(nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=s,bias = bias))
    else:
        return nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1,stride=s,bias = bias)

def Linear(ch_in,ch_out,SN = False,bias = True):
    if SN:
        return SpectralNorm(nn.Linear(ch_in,ch_out).apply(init_weight))
    else:
        return nn.Linear(ch_in,ch_out).apply(init_weight)

def conv4x4(ch_in,ch_out,SN = False,s = 2,p=1,bias = True):
    if SN:
        return SpectralNorm(nn.Conv2d(ch_in, ch_out, kernel_size=4, padding=p,stride=s,bias = bias))
    else:
        return nn.Conv2d(ch_in, ch_out, kernel_size=4, padding=p,stxride=s,bias = bias)    
    
def conv1x1(ch_in,ch_out,SN = False,s = 1,p=0,bias = True):
    if SN:
        return SpectralNorm(nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=p,stride=s,bias = bias))
    else:
        return nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=p,stride=s,bias = bias)

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
            
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1,0.02)
        m.bias.data.zero_()
    
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    
    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        
        
class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_condition,SN=False,cond_method='cbn'):
        super().__init__()
        self.n_condition = n_condition # number of classes
        self.cond_method = cond_method
        self.in_channel = in_channel
        #self.fix_scale = fix_scale # whether to fix the scalling param for all conditions or not.

        self.bn = nn.BatchNorm2d(in_channel, affine=False)  # no learning parameters

        if self.cond_method == 'cbn_fix_scale':
            out_channels = in_channel
            self.embed_gamma = torch.nn.Parameter(torch.ones(1,out_channels))
            nn.init.orthogonal_(self.embed_gamma.data, gain=1)
        else:
            out_channels = in_channel*2
            
        if self.cond_method == 'conv1x1':
            self.mlp_shared = nn.Sequential(
            conv1x1(1, 128,bias = True,SN=SN),
            nn.ReLU()
            )
            self.embed = conv1x1(128, out_channels,bias = True,SN=SN)
        elif self.cond_method == 'conv3x3':
            self.mlp_shared = nn.Sequential(
            conv3x3(1, 128,bias = True,SN=SN),
            nn.ReLU()
            )
            self.embed = conv3x3(128, out_channels,bias = True,SN=SN)
        else:
            self.embed = Linear(n_condition, out_channels,bias = True,SN=SN)

        if self.cond_method == 'cbn_fix_scale':
            self.embed.weight.data.zero_()
        else:
            nn.init.orthogonal_(self.embed.weight.data[:, :in_channel], gain=1)
            self.embed.weight.data[:, in_channel:].zero_()

    def forward(self, inputs, label):
        out = self.bn(inputs)
        if self.cond_method == 'cbn_fix_scale':
            beta = self.embed(label.float())
            gamma = self.embed_gamma
            gamma = gamma.unsqueeze(2).unsqueeze(3)
            beta = beta.unsqueeze(2).unsqueeze(3)
        elif 'conv' in self.cond_method:
            label = label.view(-1,1,label.size(-1),label.size(-1))
            label = nn.Upsample(size= inputs.size(-1))(label)
            actv  = self.mlp_shared(label.float())
            embed = self.embed(actv)
            gamma, beta = embed.chunk(2, dim=1)
        else:
            embed = self.embed(label.float())
            gamma, beta = embed.chunk(2, dim=1)
            gamma = gamma.unsqueeze(2).unsqueeze(3)
            beta = beta.unsqueeze(2).unsqueeze(3)
        out = (1+gamma) * out + beta
        return out
 
class Attention(nn.Module):
    def __init__(self, channels,SN=False):
        super().__init__()
        self.channels = channels
        self.theta    = conv1x1(channels, channels//8,SN=SN).apply(init_weight)
        self.phi      = conv1x1(channels, channels//8,SN=SN).apply(init_weight)
        self.g        = conv1x1(channels, channels//2,SN=SN).apply(init_weight)
        self.o        = conv1x1(channels//2, channels,SN=SN).apply(init_weight)
        self.gamma    = nn.Parameter(torch.tensor(0.), requires_grad=True)
        
    def forward(self, inputs):
        batch,c,h,w = inputs.size()
        theta = self.theta(inputs) #->(*,c/8,h,w)
        phi   = F.max_pool2d(self.phi(inputs), [2,2]) #->(*,c/8,h/2,w/2)
        g     = F.max_pool2d(self.g(inputs), [2,2]) #->(*,c/2,h/2,w/2)
        
        theta = theta.view(batch, self.channels//8, -1) #->(*,c/8,h*w)
        phi   = phi.view(batch, self.channels//8, -1) #->(*,c/8,h*w/4)
        g     = g.view(batch, self.channels//2, -1) #->(*,c/2,h*w/4)
        
        beta = F.softmax(torch.bmm(theta.transpose(1,2), phi), -1) #->(*,h*w,h*w/4)
        o    = self.o(torch.bmm(g, beta.transpose(1,2)).view(batch,self.channels//2,h,w)) #->(*,c,h,w)
        return self.gamma*o + inputs
    
    
class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels,hidden_channels=None, upsample=False,n_classes = 0, leak = 0,SN = False,cond_method = 'cbn'):
        super(ResBlockGenerator, self).__init__()
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        
        self.upsample = upsample
        self.learnable_sc = (in_channels != out_channels) or upsample
        
        self.conv1 = conv3x3(in_channels,hidden_channels,SN).apply(init_weight)
        self.conv2 = conv3x3(hidden_channels,out_channels,SN).apply(init_weight)
        self.conv3 = conv1x1(in_channels,out_channels,SN).apply(init_weight)
        self.upsampling = nn.Upsample(scale_factor=2)
        self.n_cl = n_classes # number of classes
        if n_classes == 0:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(hidden_channels)
        else:                
            self.bn1 = ConditionalNorm(in_channels,n_classes,SN= SN,cond_method=cond_method)
            self.bn2 = ConditionalNorm(hidden_channels,n_classes,SN=SN,cond_method=cond_method)

        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU() 

    def shortcut(self, x):
        if self.learnable_sc:
            if self.upsample:
                x = self.upsampling(x)
                x = self.conv3(x)
            else:
                x = self.conv3(x)
            return x
        else:
            return x

    def forward(self, x,y=None):
        if self.n_cl >0:
            out = self.activation(self.bn1(x,y))
        else:
            out = self.activation(self.bn1(x))

        if self.upsample:
             out = self.upsampling(out)

        out = self.conv1(out)
        if self.n_cl >0:
            out = self.activation(self.bn2(out,y))
        else:
            out = self.activation(self.bn2(out))

        out = self.conv2(out)
        out_res = self.shortcut(x)
        return out + out_res

class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=False,hidden_channels=None,leak = 0,
                 SN=True,BN=False):
        super(ResBlockDiscriminator, self).__init__()
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        
        self.conv1 = conv3x3(in_channels, hidden_channels,SN = SN).apply(init_weight)
        self.conv2 = conv3x3(hidden_channels, out_channels,SN = SN).apply(init_weight)
        self.conv3 = conv1x1(in_channels, out_channels,SN = SN).apply(init_weight)

        self.learnable_sc = (in_channels != out_channels) or downsample
        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU() 

        self.downsampling = nn.AvgPool2d(2)
        self.downsample = downsample

        self.BN = BN
        if BN:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(hidden_channels)

    def residual(self, x):
        h = x
        if self.BN:
            h = self.bn1(h)
        h = self.activation(h)
        h = self.conv1(h)
        if self.BN:
            h = self.bn2(h)
        h = self.activation(h)
        h = self.conv2(h)
        if self.downsample:
            h = self.downsampling(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.conv3(x)
            if self.downsample:
                return self.downsampling(x)
            else:
                return x
        else:
            return x

    def forward (self, x):
        return self.residual(x) + self.shortcut(x)

class OptimizedBlock(nn.Module):

    def __init__(self, in_channels, out_channels,leak =0,SN=True):
        super(OptimizedBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels,SN=SN).apply(init_weight)
        self.conv2 = conv3x3(out_channels, out_channels,SN=SN).apply(init_weight)
        self.conv3 = conv1x1(in_channels, out_channels,SN=SN).apply(init_weight)
        
        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU() 
        
        self.model = nn.Sequential(
            self.conv1,
            self.activation,
            self.conv2,
            nn.AvgPool2d(2)  # stride = 2 ( default = kernel size)
        )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.conv3
        )
    def forward(self, x):
        return self.model(x) + self.bypass(x)     