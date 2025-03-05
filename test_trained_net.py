# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:51:19 2024
    let's study the effect of the mask on the resulting image
    
    1. does the probability match?
    2. how different scales influence?
    3. how multiple masks "interact"
    
@author: roberto.miele
"""
import argparse
from FMclasses import * 
import torch
import matplotlib.pyplot as plt
from models import generators
from models import generators_norne
import subprocess
import os
from Gslib import Gslib
import time
from sklearn.metrics import confusion_matrix
import numpy as np
workdir='C:/Users/rmiele/OneDrive - Universidade de Lisboa/PhD/SPADE_code/'

# C:/Users/rmiele/OneDrive - Universidade de Lisboa/PhD/SPADE_code/500_500.pth  SPADE_Synthetic_trained.pth
save_folder= 'Manoscritto/Imgs/'

parser = argparse.ArgumentParser()

parser.add_argument("--project_path", type=str, default=workdir)
parser.add_argument("--in_folder",type=str, default="/Input_Norne")
parser.add_argument("--nx",type=int, default=64)
parser.add_argument("--ny",type=int, default=1)
parser.add_argument("--nz",type=int, default=64)
parser.add_argument("--wavelet_file", type=str, default='/wavelet_simm.asc')
parser.add_argument("--W_weight",type=int, default=1/300)
parser.add_argument("--type_of_FM",type=str, default='fullstack')
parser.add_argument("--n_it", type=int, default=100)
parser.add_argument("--n_sim_f", type=int, default=50)
parser.add_argument("--n_sim_ip", type=int, default=16)
parser.add_argument("--avg_cond", type=bool, default=True, help='Frankenstein is smoothed with 2x2 kernel')
parser.add_argument("--device",type=str, default='cpu')
parser.add_argument("--saved_state", type=str, default='/trained_state/Synthetic_model_200_ema.pth')
parser.add_argument("--null_val", default=-9999.00,type=float, help='null value in well data')

args= parser.parse_args()

state = torch.load(args.project_path+args.saved_state, map_location=args.device)

netG = generators.Res_Generator(256,img_ch=1,n_classes = 1
                                ,base_ch = 64, leak = 0,att = True
                                ,SN = True
                                ,cond_method = 'conv1x1').to(args.device)


netG.load_state_dict(state['netG_state_dict'])

# %% effect of one point mask at different sizes

def changecond(cond, size):
    cond[:,:,int(32-size/2):int(32+size/2),int(32-size/2):int(32+size/2)]+=0.05
    
    return cond

fig, ax = plt.subplots(3,4, sharey='row', figsize=(10,5.5),dpi=300)
c= 0
for size in [2,4,8,16]: #different masks sizes.
    cond= torch.zeros((args.n_sim_f,1,args.nz,args.nx)).to(args.device)
    true = np.zeros((20,size*size))
    pred = np.zeros((20,size*size))
    ax[0,c].set_title(f'Mask {size} X {size}')
       
    for i in range(20): 
        z = torch.randn(args.n_sim_f,256).to(args.device)
    
        facies = (netG(z,cond).detach().cpu()+1)/2
        
        pred[i] = facies.round()[:,:,int(32-size/2):int(32+size/2),int(32-size/2):int(32+size/2)].mean(0).squeeze().flatten()
    
        true[i] = cond[0,None,:,int(32-size/2):int(32+size/2),int(32-size/2):int(32+size/2)].flatten()
    
        cond = changecond(cond, size)
        
    ax[0,c].imshow(cond[0].squeeze(), cmap='jet', vmin=0, vmax=1)
    ax[1,c].imshow(facies.squeeze().mean(0), cmap='jet', vmin=0, vmax=1)
    ax[2,c].scatter(true.flatten(), pred.flatten())
    
    lims = [0,1]

    # now plot both limits against eachother
    ax[2,c].plot(lims, lims, 'red', alpha=0.75, zorder=1)
    ax[2,c].set_aspect('equal')
    ax[2,c].set_xlim(lims)
    ax[2,c].set_ylim(lims)
    ax[0,0].set_ylabel('Y')
    for i in range(4): ax[0,i].set_xlabel('X')
    ax[1,0].set_ylabel('Simulated')
    for i in range(4): ax[2,i].set_xlabel('Probability')
    c+=1

# plt.savefig(workdir + save_folder + 'mask_prob_syn1.png',dpi=300, bbox_inches = 'tight')
plt.show()

#%% Real case
parser = argparse.ArgumentParser()

parser.add_argument("--project_path", type=str, default='C:/Users/roberto.miele/OneDrive - Universidade de Lisboa/6. SPADE')
parser.add_argument("--in_folder",type=str, default="/Input_Norne")
parser.add_argument("--nx",type=int, default=128)
parser.add_argument("--ny",type=int, default=1)
parser.add_argument("--nz",type=int, default=128)
parser.add_argument("--wavelet_file", type=str, default='/wavelet_simm.asc')
parser.add_argument("--W_weight",type=int, default=1/300)
parser.add_argument("--type_of_FM",type=str, default='fullstack')
parser.add_argument("--n_it", type=int, default=100)
parser.add_argument("--n_sim_f", type=int, default=30)
parser.add_argument("--n_sim_ip", type=int, default=16)
parser.add_argument("--avg_cond", type=bool, default=True, help='Frankenstein is smoothed with 2x2 kernel')
parser.add_argument("--device",type=str, default='cpu')
parser.add_argument("--saved_state", type=str, default='/500_390.pth')
parser.add_argument("--null_val", default=-9999.00,type=float, help='null value in well data')

args= parser.parse_args()
state = torch.load(args.project_path+args.saved_state, map_location=args.device)
netG = generators_norne.Res_Generator(state['args'].zdim,img_ch=1,n_classes = 1
                                ,base_ch = state['args'].G_ch, leak = 0,att = True
                                ,SN = False
                                ,cond_method = 'conv1x1').to(args.device)

netG.load_state_dict(state['netG_state_dict'])
# %% effect of one point mask at different sizes

def changecond(cond, size):
    cond[:,:,int(64-size/2):int(64+size/2),int(64-size/2):int(64+size/2)]+=0.05
    
    return cond

fig, ax = plt.subplots(3,4, sharey='row', figsize=(10,5.5),dpi=300)
c= 0
for size in [2,4,8,16]: #different masks sizes.
    cond= torch.zeros((args.n_sim_f,1,args.nz,args.nx)).to(args.device)
    true = np.zeros((20,size*size))
    pred = np.zeros((20,size*size))
    ax[0,c].set_title(f'Mask {size} X {size}')
       
    for i in range(20): 
        z = torch.randn(args.n_sim_f,state['args'].zdim).to(args.device)
    
        facies = (netG(z,cond).detach().cpu()+1)/2
        
        pred[i] = facies.round()[:,:,int(64-size/2):int(64+size/2),int(64-size/2):int(64+size/2)].mean(0).squeeze().flatten()
    
        true[i] = cond[0,None,:,int(64-size/2):int(64+size/2),int(64-size/2):int(64+size/2)].flatten()
    
        cond = changecond(cond, size)
        
    ax[0,c].imshow(cond[0].squeeze(), cmap='jet', vmin=0, vmax=1)
    ax[1,c].imshow(facies.squeeze().mean(0), cmap='jet', vmin=0, vmax=1)
    ax[2,c].scatter(true.flatten(), pred.flatten())
    
    lims = [0,1]

    # now plot both limits against eachother
    ax[2,c].plot(lims, lims, 'red', alpha=0.75, zorder=1)
    ax[2,c].set_aspect('equal')
    ax[2,c].set_xlim(lims)
    ax[2,c].set_ylim(lims)
    ax[0,0].set_ylabel('Y')
    for i in range(4): ax[0,i].set_xlabel('X')
    ax[1,0].set_ylabel('Simulated')
    for i in range(4): ax[2,i].set_xlabel('Probability')
    c+=1
plt.savefig(workdir + save_folder + 'mask_prob_Norne.png',dpi=300, bbox_inches = 'tight')
