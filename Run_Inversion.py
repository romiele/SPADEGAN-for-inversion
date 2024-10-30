# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:49:59 2024
    DA PROVARE
    
    simil<0=0
    
    media simil facies 0
    media simil facies 1
    select random
    
    weighted average Ip by simils
    
    average simil
@author: roberto.miele
"""

import argparse
from FMclasses import * 
import torch
import matplotlib.pyplot as plt
from models import generators
import os
from Gslib import Gslib
import time
import json


parser = argparse.ArgumentParser()

parser.add_argument("--project_path", type=str, default='C:/[...]/')
parser.add_argument("--in_folder",type=str, default="/Input_synthetic_case")
parser.add_argument("--outdir",type=str, default="/output")
parser.add_argument("--seismic_data", type=str, default='/Input_synthetic_case/Real_seismic_[...].out')
parser.add_argument("--well_data", type=str, default='/well_data')
parser.add_argument("--Ip_model", type=int, default=None, help='fill only if you want to modify an existing seismic data from the synthetic examples')
parser.add_argument("--nx",type=int, default=64)
parser.add_argument("--ny",type=int, default=1)
parser.add_argument("--nz",type=int, default=64)
parser.add_argument("--wavelet_file", type=str, default='/wavelet_near_235ms_statistical.asc')
parser.add_argument("--W_weight",type=int, default=1)
parser.add_argument("--type_of_FM",type=str, default='fullstack')
parser.add_argument("--n_it", type=int, default=50)
parser.add_argument("--n_sim_f", type=int, default=100)
parser.add_argument("--n_sim_ip", type=int, default= 1)
parser.add_argument("--caps", type=bool, default= True, help='avoids possible local minima in the models (n_it must be > 4')
parser.add_argument("--type_of_corr", type=str, default='Similarity', help='pearson / Similarity / Quasi-corr')
parser.add_argument("--avg_cond", type=bool, default=False, help='Frankenstein is smoothed with 2x2 kernel')
parser.add_argument("--device",type=str, default='cpu')
parser.add_argument("--saved_state", type=str, default='/training_ckpt/500_500.pth')
parser.add_argument("--null_val", default=-9999.00,type=float, help='null value in well data')
parser.add_argument("--var_N_str", default=[1,1],type=int, help='number of variogram structures per facies [fac0, fac1,...]')
parser.add_argument("--var_nugget", default=[0,0],type=float, help='variogram nugget per facies [fac0, fac1,...]')
parser.add_argument("--var_type", default=[[1],[1]], type=int, help='variogram type per facies [fac0[str1,str2,...], fac1[str1,str2,...],...]: 1=spherical,2=exponential,3=gaussian')
parser.add_argument("--var_ang", default=[[0,0],[0,0]], type=float, help='variogram angles per facies [fac0[angX,angZ], fac1[angX,angZ],...]')
parser.add_argument("--var_range", default=[[[30,10]],[[40,40]]], type=float, help='variogram ranges per structure and per facies [fac0[str0[rangeX,rangeZ],str1[rangeX,rangeZ]...],fac1[str1[rangeX,rangeZ],...],...]')
parser.add_argument("--cond_ip", default= True , type=bool, help='True if iterations>1 use local conditioning of Ip (coDSS)')
parser.add_argument("--N_layers", default= 30 , type=int)
args= parser.parse_args()

if not os.path.isdir(args.project_path+args.outdir):
    os.mkdir(args.project_path+args.outdir)
    os.mkdir(args.project_path+args.outdir+'/dss')

with open(args.project_path+args.outdir+'/args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# Start
print("Starting inversion")
stime= time.time()

state = torch.load(args.project_path+args.saved_state, map_location=args.device)

netG = generators.Res_Generator(state['args'].zdim,img_ch=1,n_classes = 1
                                ,base_ch = state['args'].G_ch, leak = 0,att = True
                                ,SN = False
                                ,cond_method = 'conv1x1').to(args.device)

netG.load_state_dict(state['netG_state_dict'])

Ip = ElasticModels(args)
Seis = ForwardModeling(args)
Seis.load_wavelet(args)

# load well
wells = Gslib().Gslib_read(args.project_path+args.in_folder+args.well_data).data
wells_Ip = wells[wells.columns[-2]].values
wells_F = wells[wells.columns[-1]].values


Ip0= wells[wells[wells.columns[-1]]==0]
Ip1= wells[wells[wells.columns[-1]]==1]

del wells_Ip, wells_F

if args.cond_ip==False:
    #avoids local conditioning from well data
    Ip0.i= np.random.randint(2,3000,len(Ip0))
    Ip0.j= np.random.randint(2,3000,len(Ip0))
    Ip0.k= np.random.randint(2,3000,len(Ip0))
    
    Ip1.i= np.random.randint(2,3000,len(Ip1))
    Ip1.j= np.random.randint(2,3000,len(Ip1))
    Ip1.k= np.random.randint(2,3000,len(Ip1))

Gslib().Gslib_write('Ip_zone0', ['x','y','z','Ip'], Ip0, 4,1,len(Ip0), args.project_path+args.in_folder )
Gslib().Gslib_write('Ip_zone1', ['x','y','z','Ip'], Ip1, 4,1,len(Ip1), args.project_path+args.in_folder )

Ip0= Ip0.Ip.values
Ip1= Ip1.Ip.values

Ip.ipmin= Ip1.min()
Ip.ipmax= Ip0.max()
Ip.ipzones={0:np.array([Ip0.min(),Ip0.max()]),
            1:np.array([Ip1.min(),Ip1.max()])}

if args.Ip_model:
    #read real facies data
    rf= Gslib().Gslib_read(args.project_path+args.in_folder+f'/Real_Facies_{args.Ip_model}.out').data.values.reshape(80,100)[None,None,:64,36:]

    #read real Ip
    rip= torch.tensor(np.reshape(
        Gslib().Gslib_read(args.project_path+args.in_folder+f'/Real_Ip{args.Ip_model}_1.out').data.values.flatten(), 
        (args.nx, args.nz))) 

    #de-comment this if you want a new IP distribution
    #Ip.run_dss(torch.tensor(rf), 0, args)
    
    Gslib().Gslib_write(
        f'/Real_Ip{args.Ip_model}_1', ['Ip'], 
        rip.detach().cpu().numpy().flatten(), 
        args.nx, 1, args.nz, args.project_path+args.in_folder)
    
    Gslib().Gslib_write(f'/Real_Ip{args.Ip_model}_1', ['Ip'], 
                        rip.detach().cpu().numpy().flatten(), 
                        args.nx, 1, args.nz, args.project_path+args.outdir)

    rip = Ip.simulations = torch.Tensor(Gslib().Gslib_read(
        args.project_path+args.in_folder+'/Real_Ip'+args.seismic_data[-5:-4]+'_1.out'
        ).data.values.reshape(64,64)[None,None,:])

    Seis.real_seismic = Seis.calc_synthetic(Ip.simulations).clone()
    
    rseis= Gslib().Gslib_write(
        f'/Real_seismic_DSS{args.Ip_model}', ['Seis'], 
        Seis.real_seismic.detach().cpu().numpy().flatten(), 
        args.nx, 1, args.nz, args.project_path+args.in_folder)


cond= torch.rand(args.n_sim_f,1,8,8).to(args.device)
cond= torch.nn.Upsample(scale_factor=8).to(args.device)(cond)

if args.cond_ip: 
    cond[:,:,:,wells.i_index.values[0]-1][:,:,wells.k_index.values-1]= (torch.tensor(wells[wells.columns[-1]].values).float()) #*-1

if args.avg_cond:
    mean_conv = torch.nn.Conv2d(1, 1, kernel_size=2, stride=2).to(args.device)
    
    weights = torch.ones((2,2))/4
    
    mean_conv.weight.data = torch.FloatTensor(weights).view(1, 1, 2, 2).to(args.device)
    mean_conv.bias= None
    
    cond= mean_conv(cond)

log= torch.zeros(args.n_it,2)
flog= open(args.project_path+args.outdir+'/log.txt','w')
flog.write(f"Glob similarity (mean), Glob similarity (std dev) [num of samples={args.n_sim_f}\n")

maxglob = 0
condition = cond.clone()
best_rho_it = torch.zeros(args.nx, args.nz)
best_facies_it = torch.zeros(args.nx, args.nz)
best_ip_it = torch.zeros(args.nx, args.nz)

for i in range(args.n_it):
    
    z= torch.randn(args.n_sim_f,state['args'].zdim).to(args.device)
    
    facies= netG(z,cond).detach()
    facies[facies<=0]=-1
    facies[facies>0]=1
    
    if 'DSS' in args.outdir: 
        Ip.run_dss(facies, i, args)
    else: Ip.det_Ip(facies)
    
    Seis.calc_synthetic(Ip.simulations)
    Seis.check_seis_distance(args) #getting misfit
    # Seis.misfit = torch.round(Seis.misfit, decimals=5)
    facies[facies<=0]=-1
    facies[facies>0]=1
    
    best_facies = torch.zeros(args.nx, args.nz)
    best_ip = torch.zeros(args.nx, args.nz)
    
    best_rho = torch.tensor(np.amax(Seis.misfit.squeeze().numpy(), axis=0))
    condiz = best_rho>0#q=best_rho_it#0
    idx = np.argmax(Seis.misfit.squeeze().numpy(), axis=0)
    for j in range(args.nx):
        for k in range(args.nz):
            best_facies[j,k] = facies[idx[j,k],0,j,k]
            best_ip[j,k] = Ip.simulations[idx[j,k],0,j,k]
    
    best_rho_it[condiz] = best_rho[condiz]
    best_facies_it[condiz] = best_facies[condiz]
    best_ip_it[condiz] = best_ip[condiz]
            
    Seis.check_seis_global_distance(args)
    
    del facies, Ip.simulations
    cap = 1
    
    if args.caps:
        if i < 1 : cap = 0.65
        elif i < 2 : cap = 0.7
        elif i < 3 : cap = 0.75
        elif i < 4 : cap = 0.8
        best_rho_it[best_rho_it>cap]= cap
                
    condition = ((best_rho_it*best_facies_it)+1).to(args.device)*0.5
    if args.cond_ip: 
        condition[cond_well.k_index.values-1,cond_well.i_index.values[0]-1] = torch.tensor(cond_well[args.Ip_Fac_lognames[1]].values).float()
        
        best_rho_it[cond_well.k_index.values-1,cond_well.i_index.values[0]-1] = 1
       
        best_ip_it[cond_well.k_index.values-1,cond_well.i_index.values[0]-1] = torch.Tensor(cond_well.Ip.values)
   
    if  torch.mean(Seis.glob_misfit)>maxglob: 
        maxglob = torch.mean(Seis.glob_misfit).item()
        
        Gslib().Gslib_write('Facies_patchwork_best',['facies'], best_facies_it.detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.project_path+args.outdir)
        Gslib().Gslib_write('Similarity_patchwork_best',['similarity'], best_rho_it.detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.project_path+args.outdir)
        Gslib().Gslib_write('Facies_probability_best',['probability'], cond[0,0].detach().cpu().numpy().flatten(), cond.shape[-1], 1, cond.shape[-1], args.project_path+args.outdir)
        Gslib().Gslib_write('aux_simil_best',['simil'], best_rho_it.squeeze().detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.project_path+args.outdir)
        Gslib().Gslib_write('aux_ip_best',['Ip'], best_ip_it.detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.project_path+args.outdir)
                    
    cond = condition[None,None,:] 
    if args.avg_cond:
        cond = mean_conv(cond)
    
    Gslib().Gslib_write('aux_simil',['simil'], best_rho_it.squeeze().detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.project_path+args.outdir)
    Gslib().Gslib_write('aux_ip',['Ip'], best_ip_it.detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.project_path+args.outdir)

    curglob = torch.mean(Seis.glob_misfit)
    stdglob = torch.std(Seis.glob_misfit)
    print(f"Iteration {i+1}, Average Misfit= {curglob:.3}, Std= {stdglob:.3}")
    log[i]= torch.tensor([curglob,stdglob])
    flog.write(f"{','.join(log[i].numpy().astype(str))}\n")
    
Gslib().Gslib_write('Facies_patchwork',['facies'], best_facies.detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.project_path+args.outdir)
Gslib().Gslib_write('Similarity_patchwork',['similarity'], best_rho.detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.project_path+args.outdir)
Gslib().Gslib_write('Facies_probability',['probability'], condition.detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.project_path+args.outdir)
    
flog.close()

