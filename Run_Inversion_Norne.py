# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 19:15:53 2024

@author: roberto.miele
"""
import argparse
from FMclasses import * 
import torch
import matplotlib.pyplot as plt
from models import generators_norne
import os
from Gslib import Gslib
import time
import json
import numpy as np

for condwell in [True,False]:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--project_path", type=str, default='C:/Users/rmiele/OneDrive - Universidade de Lisboa/PhD/SPADE_code/')
    parser.add_argument("--in_folder",type=str, default='/Input_Norne')
    parser.add_argument("--outdir",type=str, default=f'/output_Norne_Well{condwell}')
    
    parser.add_argument("--nx",type=int, default=128)
    parser.add_argument("--ny",type=int, default=1)
    parser.add_argument("--nz",type=int, default=128)
    
    parser.add_argument("--seismic_data", type=str, default='/fullstack')
    parser.add_argument("--type_of_FM",type=str, default='fullstack')
    parser.add_argument("--well_name", type=str, default='/Ip_facies_well3.gslib')
    parser.add_argument("--Ip_Fac_lognames", type=str, default=['Ip','Facies'])
    parser.add_argument("--wavelet_file", type=str, default='/wavelet_near.asc') #wavelet_near_235ms *10000*0.85
    parser.add_argument("--W_weight",type=int, default=1)
    
    parser.add_argument("--n_it", type=int, default=30)
    parser.add_argument("--n_sim_f", type=int, default=150)
    parser.add_argument("--n_sim_ip", type=int, default= 1)
    
    parser.add_argument("--device",type=str, default='cpu')
    parser.add_argument("--saved_state", type=str, default='/trained_state/Norne_model_200_ema.pth')
    parser.add_argument("--zdim", type=int, default=256, help='dimensionality of the latent')
    parser.add_argument("--ch", type=int, default=32, help='base channel of model')
    
    parser.add_argument("--null_val", default=-9999.00,type=float, help='null value in well data')
    parser.add_argument("--var_N_str", default=[1,1],type=int, help='number of variogram structures per facies [fac0, fac1,...]')
    parser.add_argument("--var_nugget", default=[0.2,0],type=float, help='variogram nugget per facies [fac0, fac1,...]')
    parser.add_argument("--var_type", default=[[2],[2]], type=int, help='variogram type per facies [fac0[str1,str2,...], fac1[str1,str2,...],...]: 1=spherical,2=exponential,3=gaussian')
    parser.add_argument("--var_ang", default=[[0,0],[0,0]], type=float, help='variogram angles per facies [fac0[angX,angZ], fac1[angX,angZ],...]')
    parser.add_argument("--var_range", default=[[[30,24]],[[30,30]]], type=float, help='variogram ranges per structure and per facies [fac0[str0[rangeX,rangeZ],str1[rangeX,rangeZ]...],fac1[str1[rangeX,rangeZ],...],...]')
    parser.add_argument("--N_layers", default= 30, type=int)
    parser.add_argument("--condwell", default= condwell, type=bool)
    
    args= parser.parse_args()
    
    setattr(args, 'seedlist', np.random.randint(0,args.n_sim_f*10000,args.n_sim_f).astype(int).tolist())
    
    os.makedirs(args.project_path+args.outdir+'/dss', exist_ok=True)

    with open(args.project_path+args.outdir+'/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
    # Start
    print("Starting inversion")
    stime= time.time()
    
    #load generator and trained parameters
    state = torch.load(args.project_path+args.saved_state, map_location=args.device)
    netG = generators_norne.Res_Generator(args.zdim, img_ch=1, n_classes = 1,
                                          base_ch = args.ch, leak = 0, att = True,
                                          SN = True, cond_method = 'conv1x1').to(args.device)
    
    netG.load_state_dict(state['netG_state_dict'])
    
    #define classes 
    Ip = ElasticModels(args)
    Seis = ForwardModeling(args)
    
    #load wavelet
    Seis.load_wavelet(args)
    
    #data loading + DATA PROCESSING NEEDED FOR 3D TRACE
    i_idx = 21 #i_index of seismic line and well
    wellidx = 1 #well_index
    
    # Load 3D seismic and select 2D profile
    Seismic_data= Gslib().Gslib_read(args.project_path+args.in_folder+args.seismic_data).data.values
    Seismic_data = Seismic_data.reshape(128,128,128) #reshape to 3D ([I, J, K] are [K, J, I] of Petrel)
    
    #select the seismic profile and flip (indexing is upside down)
    Seis.real_seismic = np.flip(Seismic_data[:,:,i_idx], axis=0).copy()
    Seis.real_seismic = torch.tensor(Seis.real_seismic)
    del Seismic_data
    
    # load well
    wells = Gslib().Gslib_read(args.project_path+args.in_folder+args.well_name).data
    wells = wells[['i_index','j_index','k_index', args.Ip_Fac_lognames[0], args.Ip_Fac_lognames[1], "Seismic(default)"]]
    
    #select conditioning well
    
    wells = wells[wells["i_index"]==i_idx+1]
    wells.loc[:,'i_index']=wells.loc[:,'j_index'] #invert i and j columns to handle 2D in DSS
    wells.loc[:,'j_index']=1 #set j index to origin
    wells= wells[wells!=-9999].dropna()

    #write to check location of seismic and well on sgems
    Gslib().Gslib_write('conditioning_well', ['x','y','z','Ip','Facies'], 
                        wells[['i_index','j_index','k_index',
                                   args.Ip_Fac_lognames[0], args.Ip_Fac_lognames[1]]],
                        5,1,len(wells), args.project_path+args.in_folder)
    
    plt.figure()
    plt.plot(wells['k_index']-1,wells['Seismic(default)'],label='Real seismic')
    plt.plot(wells['k_index']-1,
             Seis.calc_synthetic(torch.tensor(wells.Ip.values[None,None,:,None])).squeeze()*-1,
             label='Convolution (Ip log)')
    plt.legend()
    plt.savefig(args.project_path+args.outdir+'/seisfit_well.png')            
    plt.close()
    
    Gslib().Gslib_write('seismic_section', ['Amplitudes'], 
                        Seis.real_seismic.numpy().flatten(),
                        Seis.real_seismic.numpy().shape[0],1,Seis.real_seismic.numpy().shape[1], 
                        args.project_path+args.in_folder)
    
    if not condwell:
        numb= len(wells)
        axis = np.arange(200,1100+numb)
        np.random.shuffle(axis)
        wells['i_index']= axis[:numb]
        np.random.shuffle(axis)
        wells['j_index']= axis[:numb]
        np.random.shuffle(axis)
        wells['k_index']= axis[:numb]

    #Ip histograms and bounds per facies
    Ip0 = wells[wells[args.Ip_Fac_lognames[1]]==0].reset_index()[['i_index','j_index','k_index','Ip']]
    Ip1 = wells[wells[args.Ip_Fac_lognames[1]]==1].reset_index()[['i_index','j_index','k_index','Ip']]
    Ip0 = Ip0[Ip0!=args.null_val].dropna()
    Ip1 = Ip1[Ip1!=args.null_val].dropna()
        
    Ip.ipmin= min([min(Ip1.Ip.values),min(Ip0.Ip.values)])
    Ip.ipmax= max([max(Ip1.Ip.values),max(Ip0.Ip.values)])
    Ip.ipzones={0:np.array([Ip0.Ip.values.min(),Ip0.Ip.values.max()]),
                1:np.array([Ip1.Ip.values.min(),Ip1.Ip.values.max()])}
    
                
    #write conditioning data per zones/facies                
    Gslib().Gslib_write('Ip_zone0', ['x','y','z','Ip'], Ip0, 4,1,len(Ip0), args.project_path+args.in_folder )
    Gslib().Gslib_write('Ip_zone1', ['x','y','z','Ip'], Ip1, 4,1,len(Ip1), args.project_path+args.in_folder )
    
    #just some fancy plotting here
    if condwell:
        fig, axs = plt.subplots(1,1)
        axs.imshow(Seis.real_seismic.detach().cpu().squeeze(),cmap='seismic')
        axs.plot(wells.i_index-1,wells.k_index-1, zorder=2, c='k',linestyle='--', linewidth=3)
        axs.plot(wells.i_index+wells.Ip*0.007-50,wells.k_index, zorder=2, 
                 c='darkgreen', linewidth=2, label='Ip well')
        axs.legend()
        plt.savefig(args.project_path+args.outdir+'/dobswell.png')
        plt.close()
    
    plt.figure()
    plt.hist(Ip0.Ip.values, color='black', alpha=0.7, label='Shale')
    plt.hist(Ip1.Ip.values, color='orange', alpha=0.7, label='Sands')
    plt.legend()
    plt.savefig(args.project_path+args.outdir+'/Iphist.png')
    plt.close()
    
    plt.figure()
    plt.imshow(Seis.real_seismic.detach().cpu().squeeze(),cmap='seismic')
    plt.colorbar()
    plt.savefig(args.project_path+args.outdir+'/dobs.png')
    plt.close()
    
    #prior conditioning data
    if args.n_sim_f>=100: sc_sim = args.n_sim_f//2
    else: sc_sim = args.n_sim_f
    
    cond= torch.rand(args.n_sim_f,1,args.nz,args.nx).to(args.device)    
    
    def set_wellcond(cond):
        cond[:,:,wells.k_index.values-1,wells.i_index.values[0]-1] = torch.tensor(wells[args.Ip_Fac_lognames[1]].values).float()
        return cond
    
    if args.condwell: set_wellcond(cond)

    plt.figure()
    plt.imshow(cond[0].squeeze().detach(),vmin=0,vmax=1,cmap='hot')
    plt.colorbar(label='Conditioning probability of shales')
    plt.savefig(args.project_path+args.outdir+'/probs_prior.png')
    plt.close()
    
    log= torch.zeros(args.n_it,2)
    flog= open(args.project_path+args.outdir+'/log.txt','w')
    flog.write(f"Glob similarity (mean), Glob similarity (std dev) [num of samples={args.n_sim_f}\n")
    maxglob = 0
    
    print('Starting inversion')
    for i in range(args.n_it):
        facies = torch.zeros((args.n_sim_f,1, args.nz,args.nx))
        
        for ps in range(int(args.n_sim_f/sc_sim)):
            z= torch.randn(sc_sim,args.zdim).to(args.device)
            facies[sc_sim*ps:sc_sim*ps+sc_sim]= netG(z,cond[sc_sim*ps:sc_sim*ps+sc_sim]).detach()

        facies = torch.round((facies+1)/2)
        
        Ip.run_dss(facies, i, args)

        Seis.calc_synthetic(Ip.simulations)
    
        likelihood = Seis.check_seis_distance(args)
        
        # weights = likelihood/(torch.sum(likelihood, dim=0))
        # weighted_mean = torch.sum((facies * weights), dim=0)                      #weighted mean 
        # weighted_var = torch.sum((weights*(facies-weighted_mean)**2), dim=0)      #weighted mean 
        
        where_max = torch.argmax(likelihood, dim=0)[0]                          #where is the highest likelihood
        like_max = torch.amax(likelihood, dim=0)[0]                             #which value of similarity
        
        facies_max = torch.zeros_like(where_max)                                #get the facies distribution with highest likelihood
        for j in range(args.nz):
            for k in range(args.nx):
                facies_max[j,k]=facies[where_max[j,k],0,j,k]
        
        #accept or reject that distribution based on local likelihood
        
        cond_p = cond.clone()
        cond_r = torch.zeros_like(cond_p)
        for j in range(args.n_sim_f):
            p = torch.rand((args.nz,args.nx))
            mask = p<like_max
            anti_mask= mask == False
            cond_p[j,0,mask] = facies_max[mask].float()                 #if p is below likelihood, accept the occurrence of the facies 
            cond_r[j,0,mask] = like_max[mask].float()                   #if p is below likelihood, accept the correlation coefficient for ip simulation
            cond_p[j,0,anti_mask] = 0.5                                 # everywhere else, is prior probability
            
        cond_p = torch.mean(cond_p, dim=0)
        cond_r = torch.mean(cond_r, dim=0)
        cond = torch.tile(cond_p, (args.n_sim_f, 1,1,1))
             
        best_facies_it = torch.round(cond_p)[0]
        best_rho_it = cond_r[0]
        best_ip_it = torch.zeros(args.nx, args.nz)
        for j in range(args.nx):
            for k in range(args.nz):
                best_ip_it[j,k] = Ip.simulations[where_max[j,k],0,j,k]
        
        if args.condwell: 
            set_wellcond(cond)
            best_facies_it[wells.k_index.values-1,wells.i_index.values[0]-1] = torch.Tensor(wells.Facies.values)
            best_rho_it[wells.k_index.values-1,wells.i_index.values[0]-1] = 1
            best_ip_it[wells.k_index.values-1,wells.i_index.values[0]-1] = torch.Tensor(wells.Ip.values)
            Gslib().Gslib_write('aux_simil',['simil'], best_rho_it.squeeze().detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.project_path+args.outdir)
            Gslib().Gslib_write('aux_ip',['Ip'], best_ip_it.detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.project_path+args.outdir)
        
        Seis.check_seis_global_distance(args)
        if  torch.mean(Seis.glob_misfit)>maxglob: 
            maxglob = torch.mean(Seis.glob_misfit).item()
    
            Gslib().Gslib_write('Facies_patchwork_best',['facies'], best_facies_it.detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.project_path+args.outdir)
            Gslib().Gslib_write('Similarity_patchwork_best',['similarity'], best_rho_it.detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.project_path+args.outdir)
            Gslib().Gslib_write('Facies_probability_best',['probability'], cond[0,0].detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.project_path+args.outdir)
            Gslib().Gslib_write('aux_simil_best',['simil'], best_rho_it.squeeze().detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.project_path+args.outdir)
            Gslib().Gslib_write('aux_ip_best',['Ip'], best_ip_it.detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.project_path+args.outdir)
    
        del facies, Ip.simulations
        
        curglob = torch.mean(Seis.glob_misfit)
        stdglob = torch.std(Seis.glob_misfit)
        print(f"Iteration {i+1}, Average Misfit= {curglob:.3}, Std= {stdglob:.3}")
        log[i]= torch.tensor([curglob,stdglob])
        flog.write(f"{','.join(log[i].numpy().astype(str))}\n")
        
        if (i)%10==0:
            ff = (netG(torch.randn(args.n_sim_f, 256).to(args.device), cond)+1).detach().cpu()*0.5
            meanit= torch.mean(ff,dim=0).squeeze().numpy()
            
            plt.figure()
            plt.imshow(meanit,vmin=0,vmax=1, cmap='hot')
            plt.colorbar(label='Probability of sands')
            plt.savefig(args.project_path+args.outdir+f'/probs_it_{i+1}.png')
            plt.close()
            
            plt.figure()
            plt.imshow(cond.detach().cpu().mean(0).squeeze(),vmin=0,vmax=1, cmap='jet')
            plt.colorbar(label='Conditioning probability (Sands)')
            plt.savefig(args.project_path+args.outdir+f'/Conditioning_{i+1}.png')
            plt.close()
            
            plt.figure()
            plt.errorbar(torch.arange(i+1).numpy(),log[:i+1,0].numpy(),yerr=log[:i+1,1].numpy(), color ='k')
            plt.legend('Correlation coefficient')
            plt.ylim([-0.1,1])
            plt.savefig(args.project_path+args.outdir+'/log.png')
            plt.close()
            
            if args.condwell: 
                plt.figure()
                plt.imshow(best_ip_it.detach().cpu().squeeze(),
                           cmap='jet', vmin=Ip.ipmin,vmax=Ip.ipmax)
                plt.colorbar(label= r'Conditioning $I_P$ values')
                plt.savefig(args.project_path+args.outdir+f'/it{i+1}_aux_ip.png')
                plt.close()
            
            plt.figure()
            plt.imshow(best_rho_it.detach().cpu().squeeze(),cmap='hsv',vmin=0,vmax=1)
            plt.colorbar(label= r'Highest similarity coefficients')
            plt.savefig(args.project_path+args.outdir+f'/it{i+1}_aux_simil.png')
            plt.close()
            
            plt.figure()
            plt.imshow(best_facies_it.detach().cpu().squeeze(),cmap='jet',vmin=0,vmax=1)
            plt.colorbar(label= r'Aux facies')
            plt.savefig(args.project_path+args.outdir+f'/it{i+1}_aux_fac.png')
            plt.close()
            
            plt.figure()
            plt.imshow(Seis.syn_seismic.mean(0).squeeze(), cmap='seismic')
            plt.colorbar(label='Average synthetic seismic')
            plt.savefig(args.project_path+args.outdir+f'/syn_seis{i+1}.png')
            plt.close()
            del meanit
        
    flog.close()
