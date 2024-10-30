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
from models.layers import *
import matplotlib.pyplot as plt
       

class Res_Discriminator_simple(nn.Module):
     def __init__(self, img_ch=3,base_ch = 32,n_classes=0,leak =0,att = False
         ,cond_method = 'concat',SN= True,SN_y = False):
         super(Res_Discriminator_simple, self).__init__()
        
         if leak >0:
             self.activation = nn.LeakyReLU(leak)
         else:
             self.activation = nn.ReLU()
            
         self.base_ch = base_ch
         self.att = att
         self.n_classes = n_classes # num_classes
         self.cond_method = cond_method

         #method of conditioning
         if n_classes !=0 :
             # conditioning by concatenation 
             if self.cond_method =='concat': 
                 # concatenate after the 3rd layer
                 self.embed_y = Linear(n_classes,base_ch* 2*8*8,SN=SN_y)
                    
             # conditioning by projection    
             elif self.cond_method =='proj': 
                 self.embed_y = Linear(n_classes,base_ch * 16,SN=SN_y)
                 #self.embed_y = nn.Embedding(n_classes,ch * 16).apply(init_weight)
            
             elif self.cond_method =='conv1x1':
                 self.embed_y = conv1x1(1,base_ch * 4,SN=SN_y)
             elif self.cond_method =='conv3x3':
                 self.embed_y = conv3x3(1,base_ch * 4,SN=SN_y)
                
                        
         self.block1=OptimizedBlock(img_ch, base_ch,leak = leak,SN=SN)  #64*64*32
         if att:
              self.attention = Attention(base_ch,SN=SN)    
         self.block2=ResBlockDiscriminator(base_ch, base_ch*2, downsample=True,leak = leak,SN=SN) #32*32*64
        
         self.block3=ResBlockDiscriminator(base_ch*2 , base_ch*4,downsample=True,leak = leak,SN=SN)  #16*16*128
        
         if n_classes > 0 and self.cond_method !='proj':
              self.block4=ResBlockDiscriminator(base_ch*4, base_ch*4,downsample=True,leak = leak,SN=SN)  #8*8*128
         else:    
              self.block4=ResBlockDiscriminator(base_ch*4, base_ch*8,downsample=True,leak = leak,SN=SN)  #x/2
            
         self.block5=ResBlockDiscriminator(base_ch* 8, base_ch*16,leak = leak,SN=SN) #8*8*512

         self.fc =  Linear(self.base_ch*16, 1,SN=SN) 

     def forward(self,x,y=None): 
         # fig, axs = plt.subplots(7,2, figsize=(9,23))
         # ax= axs[0,0].imshow(x[0,0].detach().cpu(), cmap='jet', interpolation='none',aspect ='auto')
         # fig.colorbar(ax)
         # axs[0,0].set_title('input')
         # axs[0,0].axis('off')
         # axs[0,1].axis('off')

         # h = self.block1(x)     
         
         # ax= axs[1,0].imshow(h[0,0].detach().cpu(), cmap='jet', interpolation='none',aspect ='auto')
         # fig.colorbar(ax)
         # axs[1,0].set_title('layer1 64x64')
         # axs[1,0].axis('off')
         # axs[1,1].axis('off')

         
         # if self.att:
         #     h = self.attention(h)     
         # ax= axs[2,0].imshow(h[0,0].detach().cpu(), cmap='jet', interpolation='none',aspect ='auto')
         # fig.colorbar(ax)
         # axs[2,0].set_title('attention 64x64')
         # axs[2,0].axis('off')
         # axs[2,1].axis('off')        
         
         # h = self.block2(h)
         # ax= axs[3,0].imshow(h[0,0].detach().cpu(), cmap='jet', interpolation='none',aspect ='auto')
         # fig.colorbar(ax)

         # axs[3,0].set_title('layer2 32x32')
         # axs[3,0].axis('off')
         # axs[3,1].axis('off')   
           
         # h = self.block3(h)
         # ax= axs[4,0].imshow(h[0,0].detach().cpu(), cmap='jet', interpolation='none',aspect ='auto')
         # fig.colorbar(ax)
         # axs[4,0].set_title('layer3 16x16')
         # axs[4,0].axis('off')
         # axs[4,1].axis('off')         
         
         # h = self.block4(h)
         # ax= axs[5,0].imshow(h[0,0].detach().cpu(), cmap='jet', interpolation='none',aspect ='auto')
         # fig.colorbar(ax)
         # axs[5,0].set_title('layer4 8x8')
         # axs[5,0].axis('off')
         # axs[5,1].axis('off')         
         
         # if y is not None and 'conv' in self.cond_method:
         #     w = h.size(-1)
         #     y = y.view(-1,1,w,w)
         #     h_y = self.embed_y(y)
         #     axs[4,1].imshow(y[0,0].detach().cpu(), vmin=0,vmax=1, cmap='jet', interpolation='none',aspect ='auto')
         #     ax= axs[5,1].imshow(h_y[0,0].detach().cpu(), cmap='jet', interpolation='none',aspect ='auto')
         #     fig.colorbar(ax)
         #     axs[5,1].set_title('embedded mask 8x8')
         #     axs[5,1].axis('off')
             
         #     h = torch.cat((h,h_y),1)
             
         # h = self.block5(h)
         # h = self.activation(h)
         
         # ax= axs[6,0].imshow(h[0,0].detach().cpu(), cmap='jet', interpolation='none',aspect ='auto')
         # fig.colorbar(ax)
         # axs[6,0].set_title('layer5 8x8')
         # axs[6,0].axis('off')
         # axs[6,1].axis('off')      
         
         # h = torch.sum(h,dim = (2,3))
         # h = h.view(-1, self.base_ch*16)
         # output = self.fc(h)
         
         # axs[6,1].set_title(torch.mean(output).item(), pad=-50)
         # plt.savefig('D:/SPADE_facies/models/Discriminator_norne_simple_fake.png', dpi=400, bbox_inches = 'tight')
         
          h = self.block1(x)
          if self.att:
              h = self.attention(h)
          h = self.block2(h)
          h = self.block3(h)
        
          if y is not None and self.cond_method =='concat':
              h_y = self.embed_y(y)
              h_y = h_y.view(-1,self.base_ch*2,8,8)
              h = torch.cat((h,h_y),1)
          #print(h.shape)    
          h = self.block4(h)
          if y is not None and 'conv' in self.cond_method:
              w = h.size(-1)
              y = y.view(-1,1,w,w)
              h_y = self.embed_y(y)
              h = torch.cat((h,h_y),1)
          h = self.block5(h)   
          h = self.activation(h)

          h = torch.sum(h,dim = (2,3))
          h = h.view(-1, self.base_ch*16)
          output = self.fc(h)

          if y is not None and self.cond_method =='proj': # use projection
              output += torch.sum(self.embed_y(y) * h, dim=1, keepdim=True)    
        
           
          return output #,psi,self.embed_y(y),h,torch.sum(self.embed_y(y) * h, dim=1, keepdim=True)


        
        
class Res_Discriminator8x8(nn.Module):
    def __init__(self, img_ch=3,base_ch = 32,n_classes=0,leak =0,att = False
        ,cond_method = 'concat',SN= True,SN_y = False):
        super(Res_Discriminator8x8, self).__init__()
        
        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU()
            
        self.base_ch = base_ch
        self.att = att
        self.n_classes = n_classes # num_classes
        self.cond_method = cond_method

        #method of conditioning
        if n_classes !=0 :
            # conditioning by concatenation 
            if self.cond_method =='concat': 
                # concatenate after the 3rd layer
                self.embed_y = Linear(n_classes,base_ch* 2*8*8,SN=SN_y)
                    
            # conditioning by projection    
            elif self.cond_method =='proj': 
                self.embed_y = Linear(n_classes,base_ch * 16,SN=SN_y)
                #self.embed_y = nn.Embedding(n_classes,ch * 16).apply(init_weight)
            
            elif self.cond_method =='conv1x1':
                self.embed_y = conv1x1(1,base_ch * 4,SN=SN_y)
            elif self.cond_method =='conv3x3':
                self.embed_y = conv3x3(1,base_ch * 4,SN=SN_y)
                        
        self.block1=OptimizedBlock(img_ch, base_ch,leak = leak,SN=SN)  
        if att:
             self.attention = Attention(base_ch,SN=SN)             
        self.block2=ResBlockDiscriminator(base_ch, base_ch*2, downsample=True,leak = leak,SN=SN) 
                     
        self.block3=ResBlockDiscriminator(base_ch*2 , base_ch*4,downsample=True,leak = leak,SN=SN)  

        if n_classes > 0 and self.cond_method != 'proj':
            self.block4 = ResBlockDiscriminator(base_ch * 4, base_ch * 4, downsample=True, leak=leak, SN=SN)
        else:
            self.block4 = ResBlockDiscriminator(base_ch * 4, base_ch * 8, downsample=True, leak=leak, SN=SN)
        
        # Additional block for 128x128 input maintained dimensionality
        self.block5 = ResBlockDiscriminator(base_ch * 8, base_ch * 16, leak=leak, SN=SN)
        
        # Final block with downsampling 
        self.block6 = ResBlockDiscriminator(base_ch * 16, base_ch * 16, downsample=True, leak=leak, SN=SN)
        self.fc = Linear(self.base_ch * 16, 1, SN=SN)

    def forward(self,x,y=None):
        # fig, axs = plt.subplots(8,2, figsize=(9,25))
        # ax =axs[0,0].imshow(x[0,0].detach().cpu(),  cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        # axs[0,0].set_title('input')
        # axs[0,0].axis('off')
        # axs[0,1].axis('off')

        h = self.block1(x)     
        
        # ax =axs[1,0].imshow(h[0,0].detach().cpu(), cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        
        # axs[1,0].set_title('layer1 64x64')
        # axs[1,0].axis('off')
        # axs[1,1].axis('off')

        
        if self.att:
            h = self.attention(h)     
        # ax =axs[2,0].imshow(h[0,0].detach().cpu(), cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)

        # axs[2,0].set_title('attention 64x64')
        # axs[2,0].axis('off')
        # axs[2,1].axis('off')        
        
        h = self.block2(h)
        # ax =axs[3,0].imshow(h[0,0].detach().cpu(), cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        # axs[3,0].set_title('layer2 32x32')
        # axs[3,0].axis('off')
        # axs[3,1].axis('off')   
          
        h = self.block3(h)
        # ax =axs[4,0].imshow(h[0,0].detach().cpu(),cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        # axs[4,0].set_title('layer3 16x16')
        # axs[4,0].axis('off')
        # axs[4,1].axis('off')         
        
        h = self.block4(h)
        # ax =axs[5,0].imshow(h[0,0].detach().cpu(),  cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        # axs[5,0].set_title('layer4 8x8')
        # axs[5,0].axis('off')
        # axs[5,1].axis('off')         
        
        if y is not None and 'conv' in self.cond_method:
            w = h.size(-1)
            y = y.view(-1,1,w,w)
            h_y = self.embed_y(y)
            # ax =axs[5,1].imshow(h_y[0,0].detach().cpu(), cmap='jet', interpolation='none',aspect ='auto')
            # fig.colorbar(ax)
            # ax =axs[4,1].imshow(y[0,0].detach().cpu(), cmap='jet', interpolation='none',aspect ='auto')
            # fig.colorbar(ax)
            # axs[5,1].set_title('embedded mask 8x8')
            # axs[4,1].set_title('mask 8x8')
            # axs[5,1].axis('off')
            
            h = torch.cat((h,h_y),1)
            
        h = self.block5(h)
        # ax =axs[6,0].imshow(h[0,0].detach().cpu(), cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        # axs[6,0].set_title('layer5 4x4')
        # axs[6,0].axis('off')
        # axs[6,1].axis('off')      
        
        h = self.block6(h)
        # ax =axs[7,0].imshow(h[0,0].detach().cpu(),  cmap='jet', interpolation='none',aspect ='auto')
        # fig.colorbar(ax)
        # axs[7,0].set_title('layer6 4x4')
        # axs[7,0].axis('off')
        # axs[7,1].axis('off')         
        # plt.tight_layout()
        # plt.subplots_adjust(hspace=0.6, wspace=0.15)
        # plt.savefig('D:/SPADE_facies/models/Discriminator_norne_8x8.png', dpi=400, bbox_inches = 'tight')
        h = self.activation(h)

        h = torch.sum(h,dim = (2,3))
        h = h.view(-1, self.base_ch*16)
        
        output = self.fc(h)
        
        if y is not None and self.cond_method =='proj': # use projection
            output += torch.sum(self.embed_y(y) * h, dim=1, keepdim=True)    
        
           
        return output #,psi,self.embed_y(y),h,torch.sum(self.embed_y(y) * h, dim=1, keepdim=True)
        
        

class Res_Discriminator4x4(nn.Module):
    def __init__(self, img_ch=3,base_ch = 32,n_classes=0,leak =0,att = False
        ,cond_method = 'concat',SN= True,SN_y = False):
        super(Res_Discriminator4x4, self).__init__()
        
        if leak >0:
            self.activation = nn.LeakyReLU(leak)
        else:
            self.activation = nn.ReLU()
            
        self.base_ch = base_ch
        self.att = att
        self.n_classes = n_classes # num_classes
        self.cond_method = cond_method

        #method of conditioning
        if n_classes !=0 :
            # conditioning by concatenation 
            if self.cond_method =='concat': 
                # concatenate after the 3rd layer
                self.embed_y = Linear(n_classes,base_ch* 2*8*8,SN=SN_y)
                    
            # conditioning by projection    
            elif self.cond_method =='proj': 
                self.embed_y = Linear(n_classes,base_ch * 16,SN=SN_y)
                #self.embed_y = nn.Embedding(n_classes,ch * 16).apply(init_weight)
            
            elif self.cond_method =='conv1x1':
                self.embed_y = conv1x1(1,base_ch * 8,SN=SN_y)
            elif self.cond_method =='conv3x3':
                self.embed_y = conv3x3(1,base_ch * 4,SN=SN_y)
                        
        self.block1=OptimizedBlock(img_ch, base_ch,leak = leak,SN=SN)  #64*64*32
        if att:
             self.attention = Attention(base_ch,SN=SN)
             
        self.block2=ResBlockDiscriminator(base_ch, base_ch*2, downsample=True,leak = leak,SN=SN) #32*32*64
        
        self.block3=ResBlockDiscriminator(base_ch*2 , base_ch*4,downsample=True,leak = leak,SN=SN)  #16*16*128
        
        self.block4=ResBlockDiscriminator(base_ch*4, base_ch*8,downsample=True,leak = leak,SN=SN) #8*8*256
        
        if n_classes > 0 and self.cond_method !='proj':
             self.block5=ResBlockDiscriminator(base_ch*8, base_ch*8,downsample=True,leak = leak,SN=SN)  #4*4*256 (+4*4*256)
        else:    
             self.block5=ResBlockDiscriminator(base_ch*8, base_ch*16,downsample=True,leak = leak,SN=SN)  #x/2
        
        self.block6=ResBlockDiscriminator(base_ch* 16, base_ch*32,leak = leak,SN=SN) #4*4*1024
        
        self.fc =  Linear(self.base_ch*32, 1,SN=SN) 

    def forward(self,x,y=None):
        
        h = self.block1(x)
        if self.att:
            h = self.attention(h)
        h = self.block2(h)
        h = self.block3(h)
        
        if y is not None and self.cond_method =='concat':
            h_y = self.embed_y(y)
            h_y = h_y.view(-1,self.base_ch*2,8,8)
            h = torch.cat((h,h_y),1)
        #print(h.shape)    
        h = self.block4(h)
        
        h = self.block5(h)
        
        if y is not None and 'conv' in self.cond_method:
            w = h.size(-1)
            y = y.view(-1,1,w,w)
            h_y = self.embed_y(y)

            h = torch.cat((h,h_y),1)
            
        
        h = self.block6(h)
        
        h = self.activation(h)

        h = torch.sum(h,dim = (2,3))
        h = h.view(-1, self.base_ch*32)
        output = self.fc(h)

        if y is not None and self.cond_method =='proj': # use projection
            output += torch.sum(self.embed_y(y) * h, dim=1, keepdim=True)    
        
           
        return output #,psi,self.embed_y(y),h,torch.sum(self.embed_y(y) * h, dim=1, keepdim=True)

