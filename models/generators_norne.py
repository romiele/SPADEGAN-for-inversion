# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 19:44:16 2024

@author: roberto.miele
---> Currently in alternative 3

Alternative 1 normal architecture with cond 8x8
    DISCRIMINATOR
    Conv1 64 x 64 x basech
        Attention
    Conv2 32 x 32 x basech*2
    Conv3 16 x 16 x basech*4
    Conv4 8 x 8 x basech*4 
        (+ Condition 8 x 8 x basech*4 = 8 x 8 x basech*8)
    Conv5 8 x 8 x basech*16
        activation
        sum
        linear
        
    
    GENERATOR
    Dense 8 x 8 x basech*8    
    TConv1 16 x 16 x basech*8 
    TConv2 32 x 32 x basech*4
    TConv3 64 x 64 x basech*2
        attention
    TConv4 128 x 128 x basech
        bn
        activation
        Conv3x3 128 x 128
    
    
    ---
Alternative 2 architecture with additional layer and cond 8x8
    DISCRIMINATOR
    Conv1 64 x 64 x basech
        Attention
    Conv2 32 x 32 x basech*2
    Conv3 16 x 16 x basech*4
    Conv4 8 x 8 x basech*4 
        (+ Condition 8 x 8 x basech*4 = 8 x 8 x basech*8)
    Conv5 8 x 8 x basech*16
    Conv6 4 x 4 x basech*32
        activation
        sum
        linear
        
        
    GENERATOR
    Dense 4 x 4 x basech*16  
    TConv1 8 x 8 x basech*16
    TConv2 16 x 16 x basech*8
    TConv3 32 x 32 x basech*4
    TConv4 64 x 64 x basech*2
        attention
    TConv5 128 x 128 x basech
        bn
        activation
        Conv3x3 128 x 128  
    

"""

import torch.nn as nn
import sys
import torch.nn.utils.spectral_norm as SpectralNorm
import numpy as np
import torch
import torch.nn.functional as F
try: 
    from models.layers import *
except:
    from layers import *    
import matplotlib.pyplot as plt

class Res_Generator_simple(nn.Module):
     def __init__(self,z_dim =128,img_ch=3,base_ch =64,n_classes = 0,leak = 0,att = False,SN=False,cond_method = 'cbn'):
         super(Res_Generator_simple, self).__init__()

         self.base_ch = base_ch
         self.n_classes = n_classes
         self.att = att
         self.cond_method = cond_method
        
         if self.cond_method == 'concat':
             z_dim = z_dim+n_classes
             n_classes = 0


         if leak > 0:
             self.activation = nn.LeakyReLU(leak)
         else:
             self.activation = nn.ReLU()  
        
         self.dense = Linear(z_dim, 8 * 8 * base_ch*8,SN=SN)
        
         self.block1 = ResBlockGenerator(base_ch*8, base_ch*8,upsample=True,n_classes = n_classes,leak = leak,SN=SN,cond_method=cond_method)
         self.block2 = ResBlockGenerator(base_ch*8, base_ch*4,upsample=True,n_classes = n_classes,leak = leak,SN=SN,cond_method=cond_method)
         self.block3 = ResBlockGenerator(base_ch*4, base_ch*2,upsample=True,n_classes = n_classes,leak = leak,SN=SN,cond_method=cond_method)
         if att:
             self.attention = Attention(base_ch*2,SN=SN)
         self.block4 = ResBlockGenerator(base_ch*2, base_ch,upsample=True,n_classes = n_classes,leak = leak,SN=SN,cond_method=cond_method)
        
         self.bn = nn.BatchNorm2d(base_ch)
         self.final = conv3x3(base_ch,img_ch,SN = SN).apply(init_weight)

     def forward(self, z,y=None):
        # fig, axs = plt.subplots(8,1, figsize=(5,40))
        # ax= axs[0].imshow(y[0,0].detach().cpu(), vmin=0,vmax=1, cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        # axs[0].set_title('Mask')
        # axs[0].axis('off')
        
        # axs[1].imshow(z[0,None,:].detach().cpu(), extent=(0,128,0,10), cmap='jet', vmin=-2.5,vmax=2.5)
        # axs[1].set_title('Z')
        # axs[1].axis('off')
        
        # if self.cond_method =='concat':
        #     z = torch.cat((z,y),1)
        #     y = None
        # h = self.dense(z).view(-1,self.base_ch*8, 8, 8)
        
        # ax = axs[2].imshow(h[0,0].detach().cpu(), cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        # axs[2].set_title('Dense 8x8')
        # axs[2].axis('off')
        
        # h = self.block1(h,y)
        # ax = axs[3].imshow(h[0,0].detach().cpu(),  cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        # axs[3].set_title('Layer1 16x16')
        # axs[3].axis('off')
        
        # h = self.block2(h, y)
        # ax = axs[4].imshow(h[0,0].detach().cpu(), cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        # axs[4].set_title('Layer2 32x32')
        # axs[4].axis('off')
        
        # h = self.block3(h, y)
        # ax = axs[5].imshow(h[0,0].detach().cpu(),  cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        # axs[5].set_title('Layer3 64x64')
        # axs[5].axis('off')
        # if self.att:
        #     h = self.attention(h)     
        #     ax = axs[6].imshow(h[0,0].detach().cpu(),  cmap='jet', interpolation='none',aspect ='auto')
        #     fig.colorbar(ax)
        #     axs[6].set_title('Attention 64x64')
        #     axs[6].axis('off')
            
        # h = self.block4(h,y)
        # ax = axs[7].imshow(h[0,0].detach().cpu(),  cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        # axs[7].set_title('Layer4 128x128')
        # axs[7].axis('off')
        # h = self.bn(h)
        # h = self.activation(h)
        # h = self.final(h)
        
        # plt.subplots_adjust(hspace=0.1, wspace=0.15)
        # plt.savefig('D:/SPADE_facies/models/Generator_norne_simple.png', dpi=400, bbox_inches = 'tight')
        
        if self.cond_method =='concat':
            z = torch.cat((z,y),1)
            y = None
        h = self.dense(z).view(-1,self.base_ch*8, 8, 8)
        h = self.block1(h,y)
        h = self.block2(h, y)
        h = self.block3(h, y)
        if self.att:
            h = self.attention(h)
        h = self.block4(h,y)
        h = self.bn(h)
        h = self.activation(h)
        h = self.final(h)
        return nn.Tanh()(h)
        


class Res_Generator(nn.Module):
    def __init__(self,z_dim =128,img_ch=3,base_ch =64,n_classes = 0,leak = 0,att = False,SN=False,cond_method = 'cbn'):
        super(Res_Generator, self).__init__()

        self.base_ch = base_ch
        self.n_classes = n_classes
        self.att = att
        self.cond_method = cond_method
        
        if self.cond_method == 'concat':
            z_dim = z_dim+n_classes
            n_classes = 0


        if leak > 0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU()  
        
        self.dense = Linear(z_dim, 4 * 4 * base_ch*8,SN=SN)
        
        self.block1 = ResBlockGenerator(base_ch*8, base_ch*8,upsample=True,n_classes = n_classes,leak = leak,SN=SN,cond_method=cond_method)
        self.block2 = ResBlockGenerator(base_ch*8, base_ch*4,upsample=True,n_classes = n_classes,leak = leak,SN=SN,cond_method=cond_method)
        self.block3 = ResBlockGenerator(base_ch*4, base_ch*2,upsample=True,n_classes = n_classes,leak = leak,SN=SN,cond_method=cond_method)
        
        self.block4 = ResBlockGenerator(base_ch*2, base_ch,upsample=True,n_classes = n_classes,leak = leak,SN=SN,cond_method=cond_method)
        if att:
            self.attention = Attention(base_ch,SN=SN)
        self.block5 = ResBlockGenerator(base_ch, base_ch,upsample=True,n_classes = n_classes,leak = leak,SN=SN,cond_method=cond_method)
        
        self.bn = nn.BatchNorm2d(base_ch)
        self.final = conv3x3(base_ch,img_ch,SN = SN).apply(init_weight)


    def forward(self, z,y=None):
        # fig, axs = plt.subplots(9,1, figsize=(5,40))
        # ax = axs[0].imshow(y[0,0].detach().cpu(), cmap='jet', interpolation='none',aspect ='auto')
        # plt.colorbar(ax)
        # axs[0].set_title('Mask')
        # axs[0].axis('off')
        
        # ax = axs[1].imshow(z[0,None,:].detach().cpu(), extent=(0,128,0,10))
        # axs[1].set_title('Z')
        # axs[1].axis('off')

        if self.cond_method =='concat':
            z = torch.cat((z,y),1)
            y = None
        h = self.dense(z).view(-1,self.base_ch*8, 4, 4)
        
        # ax = axs[2].imshow(h[0,0].detach().cpu(),  cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        # axs[2].set_title('Dense 4x4')
        # axs[2].axis('off')

        h = self.block1(h,y)
        # ax = axs[3].imshow(h[0,0].detach().cpu(),  cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        # axs[3].set_title('Layer1 8x8')
        # axs[3].axis('off')
        
        h = self.block2(h, y)
        # ax = axs[4].imshow(h[0,0].detach().cpu(),  cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        # axs[4].set_title('Layer2 16x16')
        # axs[4].axis('off')
        
        h = self.block3(h, y)
        # ax = axs[5].imshow(h[0,0].detach().cpu(),  cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        # axs[5].set_title('Layer3 32x32')
        # axs[5].axis('off')
        
        h = self.block4(h,y)
        # ax = axs[6].imshow(h[0,0].detach().cpu(),  cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        # axs[6].set_title('Layer4 64x64')
        # axs[6].axis('off')
        
        if self.att:
            h = self.attention(h)     
            # ax = axs[7].imshow(h[0,0].detach().cpu(),  cmap='jet', interpolation='none',aspect ='auto')
            # fig.colorbar(ax)
            # axs[7].set_title('Attention 64x64')
            # axs[7].axis('off')
            
        h = self.block5(h,y)
        # ax = axs[8].imshow(h[0,0].detach().cpu(),  cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        # axs[8].set_title('Layer5 128x128')
        # axs[8].axis('off')
        # plt.subplots_adjust(hspace=0.1, wspace=0.15)
        # plt.savefig('D:/SPADE_facies/models/Generator_norne_8x8.png', dpi=400, bbox_inches = 'tight')

        h = self.bn(h)
        h = self.activation(h)
        h = self.final(h)

        return nn.Tanh()(h)

# Testing architecture
# G= Res_Generator(img_ch=1, n_classes = 1, att=True, SN=True, cond_method='conv1x1',real=True)
# noise = torch.randn(1, 128)
# y= torch.zeros(4,4)+0.5
# fake = G(noise,y)
# print(fake.size())
