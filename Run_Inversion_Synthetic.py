# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:49:59 2024
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
import numpy as np
from scipy.stats import gaussian_kde


for typ in ['DSS']:
    for modello in [3,4,5]:
        parser = argparse.ArgumentParser()
        
        parser.add_argument("--project_path", type=str, default='C:/Users/rmiele/Work/Inversion/SPADEGAN/')
        parser.add_argument("--in_folder",type=str, default="/Input_synthetic")
        parser.add_argument("--seismic_data", type=str, default=f'/Real_seismic_{typ}{modello}.out')
        parser.add_argument("--outdir", type=str, default=f'C:/Users/rmiele/OneDrive - Université de Lausanne/Codes/Inversion/SPADEGAN/Save/')
        
        parser.add_argument("--nx",type=int, default=64)
        parser.add_argument("--ny",type=int, default=1)
        parser.add_argument("--nz",type=int, default=64)
        parser.add_argument("--wavelet_file", type=str, default='/wavelet_near_235ms_statistical.asc')
        parser.add_argument("--W_weight",type=int, default=1)
        parser.add_argument("--type_of_FM",type=str, default='fullstack')
        parser.add_argument("--n_it", type=int, default=101)
        parser.add_argument("--n_sim_f", type=int, default=150)
        parser.add_argument("--n_sim_ip", type=int, default= 1)
        parser.add_argument("--type_of_corr", type=str, default='Similarity', help='pearson / Similarity / Quasi-corr')
        parser.add_argument("--avg_cond", type=bool, default=False, help='Frankenstein is smoothed with 2x2 kernel')
        parser.add_argument("--device",type=str, default='cpu')
        parser.add_argument("--saved_state", type=str, 
                            default='C:/Users/rmiele/OneDrive - Université de Lausanne/Codes/Modeling/SPADEGAN/Save/retrain_synthetic/500_old.pth') #SPADE_Synthetic_trained.pth
        parser.add_argument("--null_val", default=-9999.00,type=float, help='null value in well data')
        parser.add_argument("--var_N_str", default=[1,1],type=int, help='number of variogram structures per facies [fac0, fac1,...]')
        parser.add_argument("--var_nugget", default=[0,0],type=float, help='variogram nugget per facies [fac0, fac1,...]')
        parser.add_argument("--var_type", default=[[1],[1]], type=int, help='variogram type per facies [fac0[str1,str2,...], fac1[str1,str2,...],...]: 1=spherical,2=exponential,3=gaussian')
        parser.add_argument("--var_ang", default=[[0,0],[0,0]], type=float, help='variogram angles per facies [fac0[angX,angZ], fac1[angX,angZ],...]')
        parser.add_argument("--var_range", default=[[[30,10]],[[40,40]]], type=float, help='variogram ranges per structure and per facies [fac0[str0[rangeX,rangeZ],str1[rangeX,rangeZ]...],fac1[str1[rangeX,rangeZ],...],...]')
        parser.add_argument("--N_layers", default= 30, type=int)
        args= parser.parse_args()
        
        args.outdir += args.seismic_data[:-4]
        os.makedirs(args.outdir+'/dss', exist_ok=True)
        
        with open(args.outdir+'/args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        
        # Start
        print("Starting inversion")
        stime= time.time()
        
        state = torch.load(args.saved_state, map_location=args.device)
        
        netG = generators.Res_Generator(256,img_ch=1,n_classes = 1
                                        ,base_ch = 64, leak = 0,att = True
                                        ,SN = False, cond_method = 'conv1x1').to(args.device)
        
        netG.load_state_dict(state['netG_state_dict'])
        
        Ip = ElasticModels(args)
        Seis = ForwardModeling(args)
        Seis.load_wavelet(args)
            
        Ip0= Gslib().Gslib_read(args.project_path+args.in_folder+'/ip_shales.out').data
        Ip1= Gslib().Gslib_read(args.project_path+args.in_folder+'/ip_sands.out').data
        
        plt.figure()
        plt.hist(Ip0.ip.values)
        plt.hist(Ip1.ip.values)
        plt.show()
        
        Ip0.i= np.random.randint(2,3000,len(Ip0))
        Ip0.j= np.random.randint(2,3000,len(Ip0))
        Ip0.k= np.random.randint(2,3000,len(Ip0))
        
        Ip1.i= np.random.randint(2,3000,len(Ip1))
        Ip1.j= np.random.randint(2,3000,len(Ip1))
        Ip1.k= np.random.randint(2,3000,len(Ip1))
        
        Gslib().Gslib_write('Ip_zone0', ['x','y','z','Ip'], Ip0, 4,1,len(Ip0), args.project_path+args.in_folder )
        Gslib().Gslib_write('Ip_zone1', ['x','y','z','Ip'], Ip1, 4,1,len(Ip1), args.project_path+args.in_folder )
        
        Ip0= Gslib().Gslib_read(args.project_path+args.in_folder+'/Ip_zone0.out').data.Ip.values
        Ip1= Gslib().Gslib_read(args.project_path+args.in_folder+'/Ip_zone1.out').data.Ip.values
        
        Ip.ipmin= Ip1.min()
        Ip.ipmax= Ip0.max()
        Ip.ipzones={0:np.array([Ip0.min(),Ip0.max()]),
                    1:np.array([Ip1.min(),Ip1.max()])}
        
        
        #building new seismic from DSS realization | facies
        rf= Gslib().Gslib_read(args.project_path+args.in_folder+'/Real_Facies_'+args.seismic_data[-5:]).data.values.reshape(80,100)[None,None,:64,36:]
        plt.figure()
        plt.imshow(rf.squeeze(),cmap='hot',vmin=0,vmax=1)            
        plt.show()
        if 'DSS' in args.seismic_data:
            
            rip= torch.tensor(np.reshape(
                Gslib().Gslib_read(args.project_path+args.in_folder+f'/Real_Ip0_1.out').data.values.flatten(), 
                (args.nx, args.nz))) #Ip.run_dss(torch.tensor(rf), 0, args)
            
            rip=Ip.run_dss(torch.tensor(rf), 0, args)
            
            plt.figure()
            plt.imshow(rip.squeeze(),cmap='jet',vmin=Ip.ipmin,vmax=Ip.ipmax)   
            plt.colorbar()
            plt.show()
        
            Gslib().Gslib_write(
                f'/Real_Ip0_1', ['Ip'], 
                rip.detach().cpu().numpy().flatten(), 
                args.nx, 1, args.nz, args.project_path+args.in_folder)
            Gslib().Gslib_write(f'/Real_Ip0_1', ['Ip'], 
                                rip.detach().cpu().numpy().flatten(), 
                                args.nx, 1, args.nz, args.outdir)
        
            rip = Ip.simulations = torch.Tensor(Gslib().Gslib_read(
                args.project_path+args.in_folder+'/Real_Ip'+args.seismic_data[-5:-4]+'_1.out'
                ).data.values.reshape(64,64)[None,None,:])
        
            Seis.real_seismic = Seis.calc_synthetic(Ip.simulations).clone()
            
            rseis= Gslib().Gslib_write(
                f'/Real_seismic_DSS0', ['Seis'], 
                Seis.real_seismic.detach().cpu().numpy().flatten(), 
                args.nx, 1, args.nz, args.project_path+args.in_folder)
        
            plt.figure()
            plt.imshow(Seis.real_seismic.squeeze(),cmap='seismic')
            plt.colorbar()
            plt.show()
        
        else: 
            Ip.ipmin= Ip1.mean()
            Ip.ipmax= Ip0.mean()
        
            rf[rf<=0]=-1
            rf[rf>0]=1
            rip= Ip.det_Ip(torch.tensor(rf))
            Seis.real_seismic = Seis.calc_synthetic(Ip.simulations).clone()
        
        plt.figure()
        plt.imshow(rip.squeeze(),cmap='jet')
        plt.colorbar()
        plt.savefig(args.outdir+'/real_ip.png')
        plt.close()
        
        plt.figure()
        plt.imshow(Seis.real_seismic.detach().cpu().squeeze(),cmap='seismic')
        plt.colorbar(label='Real seismic')
        plt.savefig(args.outdir+'/dobs.png')
        plt.close()
        
        plt.figure()
        plt.imshow(rf.squeeze(),cmap='hot')
        plt.savefig(args.outdir+'/real_Fac.png')
        plt.close()
        
        #start: each cell of 8x8 has a uniform distribution
        cond= torch.rand(args.n_sim_f,1,args.nz,args.nx).to(args.device)
        # cond= torch.nn.Upsample(scale_factor=8).to(args.device)(cond)
        
        if args.avg_cond:
            
            mean_conv = torch.nn.Conv2d(1, 1, kernel_size=2, stride=2).to(args.device)
            
            weights = torch.ones((2,2))/4
            
            mean_conv.weight.data = torch.FloatTensor(weights).view(1, 1, 2, 2).to(args.device)
            mean_conv.bias= None
            
            cond= mean_conv(cond)
        
        z= torch.randn(args.n_sim_f, 256).to(args.device)
        prior= netG(z,cond).detach().cpu()
        plt.figure()
        plt.imshow(torch.mean(prior,dim=0).squeeze().detach(),vmin=-1,vmax=1,cmap='hot')
        plt.colorbar()
        plt.savefig(args.outdir+'/prior.png')
        plt.close()
        del prior
        
        log= torch.zeros(args.n_it,2)
        flog= open(args.outdir+'/log.txt','w')
        flog.write(f"Glob similarity (mean), Glob similarity (std dev) [num of samples={args.n_sim_f}\n")
        
        maxglob = 0
        
        for i in range(args.n_it):
            z= torch.randn(args.n_sim_f, 256).to(args.device)
            
            facies= netG(z,cond).detach()
            facies = torch.round((facies+1)/2)
            
            if 'DSS' in args.seismic_data: 
                Ip.run_dss(facies, i, args)
        
            else: 
                Ip.det_Ip(facies)
            Seis.calc_synthetic(Ip.simulations)
        
            likelihood = Seis.check_seis_distance(args)
            
            weights = likelihood/(torch.sum(likelihood, dim=0))
    
            weighted_mean = torch.sum((facies * weights), dim=0)                      #weighted mean 
            weighted_var = torch.sum((weights*(facies-weighted_mean)**2), dim=0)      #weighted mean 
            
            #ALTERNATIVE 1 : accept-reject based on likelihood
            where_max = torch.argmax(likelihood, dim=0)[0]                          #where is the highest likelihood
            like_max = torch.amax(likelihood, dim=0)[0]                             #which value of similarity
            
            facies_max = torch.zeros_like(where_max)                                #get the facies distribution with highest likelihood
            for j in range(64):
                for k in range(64):
                    facies_max[j,k]=facies[where_max[j,k],0,j,k]
            
            #accept or reject that distribution based on local likelihood
            rate = min(i / (args.n_it//5), 1)
            
            like_max = like_max*rate
            
            cond_p = cond.clone()
            cond_r = torch.zeros_like(cond_p)
            for j in range(args.n_sim_f):
                p = torch.rand((64,64))
                mask = p<like_max
                anti_mask= mask == False
                cond_p[j,0,mask] = facies_max[mask].float()                 #if p is below likelihood, accept the occurrence of the facies 
                cond_r[j,0,mask] = like_max[mask].float()                   #if p is below likelihood, accept the correlation coefficient for ip simulation
                cond_p[j,0,anti_mask] = 0.5                                 # everywhere else, is prior probability
                
            cond_p = torch.mean(cond_p, dim=0)
            cond_r = torch.mean(cond_r, dim=0)
            cond = torch.tile(cond_p, (args.n_sim_f, 1,1,1))
            
            # # ALTERNATIVE 3 propose a gaussian probability centered at weighted_mean with variance being the weighted difference
            # cond = weighted_mean + torch.randn_like(likelihood)*(torch.sqrt(weighted_var))
            # cond = torch.clip(cond, 0, 1)
            
            #ALTERNATIVE 2 : fit a Gaussian kernel kde
            # for j in range(args.nx):
            #     for k in range(args.nz):
            #         kde = gaussian_kde(cond[:,0,j,k], bw_method='scott', weights = weights[:,0,j,k])
            #         new_proposal = torch.tensor(kde.resample(cond.shape[0]).squeeze())
            #         new_proposal = torch.clip(new_proposal, 0, 1)
            #         cond[:,0,j,k] = new_proposal
        
            #   ALTERNATIVE 4 use a progressive update of the prior
            # W_mean = torch.mean(likelihood, dim=0)*2   
            # cond = cond_p*(1-W_mean) + cond*(W_mean)
            
            best_facies_it = torch.round(cond_p)[0]
            best_rho_it = cond_r[0]
            best_ip_it = torch.zeros(args.nx, args.nz)
            for j in range(args.nx):
                for k in range(args.nz):
                    best_ip_it[j,k] = Ip.simulations[where_max[j,k],0,j,k]
            
            # best_ip_it = torch.sum(Ip.simulations * weights, dim=0)[0]
            # best_facies_it = np.round(torch.sum(facies * weights, dim=0)[0]) #torch.zeros(args.nx, args.nz)
            
            Seis.check_seis_global_distance(args)
            if  torch.mean(Seis.glob_misfit)>maxglob: 
                maxglob = torch.mean(Seis.glob_misfit).item()
        
                Gslib().Gslib_write('Facies_patchwork_best',['facies'], best_facies_it.detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.outdir)
                Gslib().Gslib_write('Similarity_patchwork_best',['similarity'], best_rho_it.detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.outdir)
                Gslib().Gslib_write('Facies_probability_best',['probability'], cond[0,0].detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.outdir)
                Gslib().Gslib_write('aux_simil_best',['simil'], best_rho_it.squeeze().detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.outdir)
                Gslib().Gslib_write('aux_ip_best',['Ip'], best_ip_it.detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.outdir)
        
            del facies, Ip.simulations
            
            Gslib().Gslib_write('aux_simil',['simil'], best_rho_it.squeeze().detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.outdir)
            Gslib().Gslib_write('aux_ip',['Ip'], best_ip_it.detach().cpu().numpy().flatten(), args.nx, 1, args.nz, args.outdir)
        
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
                plt.savefig(args.outdir+f'/probs_it_{i+1}.png')
                plt.close()
                
                plt.figure()
                plt.imshow(cond.detach().cpu().mean(0).squeeze(),vmin=0,vmax=1, cmap='jet')
                plt.colorbar(label='Conditioning probability (Sands)')
                plt.savefig(args.outdir+f'/Conditioning_{i+1}.png')
                plt.close()
                
                plt.figure()
                plt.errorbar(torch.arange(i+1).numpy(),log[:i+1,0].numpy(),yerr=log[:i+1,1].numpy(), color ='k')
                plt.legend('Correlation coefficient')
                plt.ylim([-0.1,1])
                plt.savefig(args.outdir+'/log.png')
                plt.close()
                
                plt.figure()
                plt.imshow(best_ip_it.detach().cpu().squeeze(),
                           cmap='jet', vmin=Ip.ipmin,vmax=Ip.ipmax)
                plt.colorbar(label= r'Conditioning $I_P$ values')
                plt.savefig(args.outdir+f'/it{i+1}_aux_ip.png')
                plt.close()
                
                plt.figure()
                plt.imshow(best_rho_it.detach().cpu().squeeze(),cmap='hsv',vmin=0,vmax=1)
                plt.colorbar(label= r'Highest similarity coefficients')
                plt.savefig(args.outdir+f'/it{i+1}_aux_simil.png')
                plt.close()
                
                plt.figure()
                plt.imshow(best_facies_it.detach().cpu().squeeze(),cmap='jet',vmin=0,vmax=1)
                plt.colorbar(label= r'Aux facies')
                plt.savefig(args.outdir+f'/it{i+1}_aux_fac.png')
                plt.close()
                
                plt.figure()
                plt.imshow(Seis.syn_seismic.mean(0).squeeze(), cmap='seismic')
                plt.colorbar(label='Average synthetic seismic')
                plt.savefig(args.outdir+f'/syn_seis{i+1}.png')
                plt.close()
                del meanit
            
            
        flog.close()
