# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:43:57 2024

@author: Roberto.Miele
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
from utils import (prepare_data, prepare_parser, 
                   prepare_device, prepare_seed, 
                   prepare_models, prepare_filename, 
                   load_from_saved, disc_2_ohe, 
                   disc_2_cont, sample_from_gen,
                   save_training_par)
import torchvision


def train(num_epochs, disc_iters, netG, netD, dataloader, optimizerD, optimizerG, 
          schedulerD, schedulerG, filename, TIME_LIMIT, start_time, 
          st_epoch,G_losses,D_losses,netG_ema=None):
    r1_loss = 0 
    if loss_fun == 'standard' or loss_fun == 'hinge':
        
        dis_criterion = nn.BCEWithLogitsLoss().to(device)
        #labels
        if args.smooth:
            label_t = 0.9
            label_f = 0
        else:
            label_t = 1
            label_f = 0
        adv_labels_t = torch.zeros((dataloader.batch_size,1)).to(device)+label_t
        adv_labels_f = torch.zeros((dataloader.batch_size,1)).to(device)+label_f
        
        def r1_penalty(D, D_real_output, real_images, gamma=10):
            # Compute gradients of discriminator's output with respect to real images
            real_images.requires_grad = True
            
            # Calculate the gradients of D_real_output with respect to real images
            gradients = torch.autograd.grad(outputs=D_real_output, inputs=real_images,
                                            grad_outputs=torch.ones_like(D_real_output),
                                            create_graph=True, retain_graph=True)[0]
            
            # Compute the L2 norm of the gradients
            gradient_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)  # L2 norm across channels, height, width
            
            # Compute the R1 penalty as the mean of the squared gradient norms
            r1_loss = 0.5 * gamma * (gradient_norm ** 2).mean()
            
            return r1_loss
    else:
        raise NotImplementedError('Not implemented for WGAN given the last changes')
    
    def update_ema(model, model_ema, alpha=args.ema_decay):
        """Update the EMA model parameters."""
        with torch.no_grad():
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.data = alpha * param_ema.data + (1 - alpha) * param.data
        return model_ema
    
    def elapsed_time(start_time):
        return time.time() - start_time

    print("Starting Training Loop...")
    # For each epoch
    lrs_G = []
    lrs_D = []
    noise_sigma= 0.3
    for epoch in range(st_epoch,num_epochs+1):
        print(epoch)
        torch.cuda.empty_cache()
        
        if TIME_LIMIT is not None and elapsed_time(start_time) > TIME_LIMIT:
            print('Time limit reached')
            break
        
        D_running_loss = 0; G_running_loss = 0; running_examples_D = 0; running_examples_G = 0

        # For each mini-batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            real_x = data[0].to(device)
            

            # label 
            if n_cl > 0:
                # discrete 0,1,...n_cl-1
                real_y = data[1].float().to(device) 
                # convert discrete values to ohe
                if args.ohe: real_y = disc_2_ohe(real_y.long(),n_cl,device)
                elif args.real_cond_list is not None: real_y = disc_2_cont(real_y,args.real_cond_list,device)
            
            else : real_y = None

            G_b_size = real_x.size(0)
            
            #instance noise
            real_x += torch.randn_like(real_x)*noise_sigma
            real_x = real_x.detach()
            
            if args.r1penalty: real_x.requires_grad = True 
            # Update D network
            for _ in range(disc_iters): 
                
                netD.zero_grad()
                # update with real labels
                real_logit = netD(real_x, real_y)
                
                # update with fake labels
                fake_x, fake_y = sample_from_gen(args,G_b_size, zdim, n_cl, netG, device, real_y=real_y)
                fake_x += torch.randn_like(fake_x)*noise_sigma
                
                fake_logit = netD(fake_x.detach(), fake_y.detach())
                
                if loss_fun == 'hinge':  
                    D_loss_fake = torch.mean(F.relu(1.0 + fake_logit))
                    D_loss_real = torch.mean(F.relu(1.0 - real_logit))
                    if args.r1penalty: r1_loss = r1_penalty(netD, real_logit, real_x)*0.5
                
                if loss_fun == 'standard':
                    D_loss_real = dis_criterion(real_logit, adv_labels_t)
                    if args.r1penalty: r1_loss = r1_penalty(netD, real_logit, real_x)
                    D_loss_fake = dis_criterion(fake_logit, adv_labels_f) 
                
                total_loss = D_loss_real+D_loss_fake+r1_loss
                
                total_loss.backward()
                optimizerD.step()
                D_running_loss += (total_loss.item()*G_b_size)/disc_iters
                
           # Update G
            netG.zero_grad()
            if not args.x_fake_GD :
                fake_x, fake_y = sample_from_gen(args, G_b_size, zdim, n_cl,
                                                 netG, device, real_y=real_y)
            
            fake_logit = netD(fake_x,fake_y)
            
            if loss_fun == 'standard':
                _G_loss = dis_criterion(fake_logit, adv_labels_t)
            
            if loss_fun == 'hinge':
                _G_loss = -torch.mean(fake_logit)
             
            _G_loss.backward()
            optimizerG.step()
            G_running_loss += _G_loss.item()*G_b_size
            
            running_examples_D+= G_b_size
            running_examples_G+= G_b_size
            
            if args.ema: netG_ema = update_ema(netG, netG_ema)
        
        if args.decay_lr:
            schedulerD.step()
            schedulerG.step()
            lrs_G.append(schedulerG.get_last_lr())
            lrs_D.append(schedulerD.get_last_lr())
            
        D_running_loss/=running_examples_D
        G_running_loss/=running_examples_G
        
        # Output training stats
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f, elapsed_time = %.4f min'
              % (epoch, num_epochs,
                 D_running_loss, G_running_loss,elapsed_time(start_time)/60))
                    
        # Save Losses for plotting later
        G_losses.append(G_running_loss)
        D_losses.append(D_running_loss)
        
        if saving_rate is not None and (epoch%saving_rate ==0 or epoch == epochs)  :

            # saving and showing results
            torch.save({
                        'epoch': epoch,
                        'netG_state_dict': netG.state_dict(),
                        'netD_state_dict': netD.state_dict(),
                        'optimizerG_state_dict': optimizerG.state_dict(),
                        'optimizerD_state_dict': optimizerD.state_dict(),
                        'netG_ema': netG_ema,
                        'Gloss':  G_losses,
                        'Dloss':  D_losses,
                        'args': args,
                        'seed': seed,
                        }, filename+str(epoch) +".pth")
            
            
        if epoch==1:
            ycond = torch.zeros((8,8))
            
            ycond[0,0] = 1
            ycond[1,1] = 0.9
            ycond[2,2] = 0.8
            ycond[3,3] = 0.7
            ycond[4,4] = 0.6
            ycond[5,5] = 0.4
            ycond[6,6] = 0.3
            ycond[7,7] = 0.15
            
            ycond= ycond.unsqueeze(0).unsqueeze(0)
            
            plt.figure()
            plt.imshow(ycond.squeeze().detach().cpu(), vmin=0,vmax=1, cmap='jet')
            plt.colorbar()
            plt.savefig(filename+"conditioning.png")
            plt.close()
        
        if epoch % 10 == 0:    
            
            imgs = netG_ema(torch.randn(100,zdim).to(device=device),
                            torch.tile(ycond,(100,1,1,1,1)).to(device=device)
                            ).squeeze().detach().cpu()
            
            mask = torch.nn.functional.interpolate(ycond,imgs.shape[-1])

            imgs = (imgs+1)/2
            
            error = imgs.mean(0).squeeze() - mask.squeeze()
            
            plt.figure()
            plt.hist(error.flatten().detach().cpu(),
                     weights=np.ones(len(error.flatten()))/len(error.flatten()))
            plt.ylim([0,1])
            plt.savefig(filename+str(epoch) +"testhist_err.png")
            plt.close()
            
            plt.figure()
            plt.imshow(imgs.mean(0), vmin=0,vmax=1, cmap='jet')
            plt.colorbar()
            plt.savefig(filename+str(epoch) +".png")
            plt.close()
            plt.figure()
            plt.imshow(imgs[0], vmin=0,vmax=1, cmap='jet')
            plt.colorbar()
            plt.savefig(filename+str(epoch) +"testre.png")
            plt.close()
            
        fig1 = plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G", c='darkblue')
        plt.plot(D_losses,label="D", c='darkred')
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        ax = plt.twinx()
        ax.plot(lrs_G, label='G lr', c='darkblue', linestyle='dashed')
        ax.plot(lrs_D, label='D lr', c='darkred', linestyle='dashed')
        ax.set_ylabel('Learning rate')
        #plt.ylim([0,6])
        plt.legend()
        fig1.savefig(filename + 'losses_lr.png')
        plt.close('all')


parser = argparse.ArgumentParser()
args = parser.parse_args()

# configurations
parser = prepare_parser()
args = parser.parse_args()

filename = prepare_filename(args)
save_training_par(args)   

# Device
device = prepare_device(args)

#Seeds
seed  = prepare_seed(args)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
    
#parameters
batch_size = args.batch_size
G_b_size = args.G_batch_size
if G_b_size is None:
    G_b_size = batch_size

disc_iters = args.disc_iters
loss_fun = args.loss
epochs = args.epochs
cgan =  args.cgan
zdim = args.zdim
img_ch = args.img_ch
saving_rate = args.save_rate

#hyperparameres
lr_D = args.lr_D
lr_G = args.lr_G
beta1 = args.beta1
beta2 = args.beta2

# TI transformations and augmentation
if args.real:
    args.transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip()])
    
else:
    # the only allowed data augmentation for synthetic channels is horizontal flipping
    args.transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomInvert(1),
        torchvision.transforms.RandomVerticalFlip(1),
        torchvision.transforms.RandomHorizontalFlip()])

dataloader = prepare_data(args)

# conditional GAN
if cgan: n_cl = args.n_cl
else: n_cl = 0

#models
netG, netD = prepare_models(args,n_cl,device)

if args.ema:
    netG_ema,_ = prepare_models(args,n_cl,device)
    
    with torch.no_grad():
        for param, param_ema in zip(netG.parameters(), netG_ema.parameters()):
            param_ema.data = param.data
            param_ema.requires_grad = False

# Optimizers 
optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, beta2))

#saved_models    
if args.saved_cp is not None:
    netG,netD,optimizerG,optimizerD,st_epoch,G_losses,D_losses = load_from_saved(args,netG,netD,optimizerG,optimizerD, device)
else:
    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    st_epoch = 1

# use decaying learning ratexs
if args.decay_lr == 'exp':
    schedulerD = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.995)
    schedulerG = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.995)
    
elif args.decay_lr == 'step':
    MILESTONES = np.linspace(30, epochs, num=5).astype(int) #None
    SCHEDULER_GAMMA_G = 0.9
    SCHEDULER_GAMMA_D = 0.8
    schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=MILESTONES, 
                                               gamma=SCHEDULER_GAMMA_D, last_epoch=-1)
    schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, milestones=MILESTONES, 
                                                gamma=SCHEDULER_GAMMA_G, last_epoch=-1)   
else: 
    schedulerD=None
    schedulerG=None
    
# Parallel GPU if ngpu > 1
if (device.type == 'cuda') and (args.ngpu > 1):
    netG = nn.DataParallel(netG, list(range(args.ngpu)))
    netD = nn.DataParallel(netD, list(range(args.ngpu)))

                
# Print the model
print(netG)
print(netD)
print("# Params. G: ", sum(p.numel() for p in netG.parameters()))
print("# Params. D: ", sum(p.numel() for p in netD.parameters()))

TIME_LIMIT = args.limit
start_time = time.time()



train(epochs,
      disc_iters,
      netG, netD,
      dataloader,
      optimizerD,
      optimizerG,
      schedulerD,
      schedulerG,
      filename,
      TIME_LIMIT,
      start_time,
      st_epoch,
      G_losses,
      D_losses,
      netG_ema)

if args.ema:
    torch.save({
                'netG_state_dict': netG_ema.state_dict(),
                }, filename+"_ema.pth")

torch.cuda.empty_cache()

