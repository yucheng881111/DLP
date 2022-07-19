# -*- coding: utf-8 -*-
"""
Created on Sat May 28 12:18:42 2022

@author: user
"""

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from models import Generator, Discriminator, weights_init
from torch.utils.data import DataLoader
from dataloader import CLEVRDataset, get_test_conditions
from evaluator import evaluation_model

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Create the generator
    netG = Generator().to(device)
    
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)
    
    # Print the model
    #print(netG)
    
    # Create the Discriminator
    netD = Discriminator().to(device)
    
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)
    
    # Print the model
    #print(netD)
    
    # Initialize BCELoss function
    criterion = nn.BCELoss()
    
    #criterion_classes = nn.BCEWithLogitsLoss()
    
    test_conditions = get_test_conditions().to(device)
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(len(test_conditions), 100, 1, 1, device=device)
    
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    
    lr = 0.0002
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    
    
    # Training Loop
    
    # Lists to keep track of progress
    img_list = []
    
    G_losses = []
    D_losses = []
    best_score = 0
    
    num_epochs = 250
    batch_size = 64
    # load training data
    dataset_train = CLEVRDataset()
    train_loader = DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=2)
    #train_iterator = iter(train_loader)
    eval_model = evaluation_model()
    
    G_iters = 5
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        g_losses = 0
        d_losses = 0
        netD.train()
        netG.train()
        # For each batch in the dataloader
        for i, (images, conditions) in enumerate(train_loader):
            '''
            try:
                images, conditions = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                images, conditions = next(train_iterator)
            '''
            images = images.to(device)
            conditions = conditions.to(device)
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            b_size = len(images)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(images, conditions)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            #errD_aux = criterion_classes(classes, conditions)
            #errD_real = errD_dis + errD_aux
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
    
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, 100, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise, conditions)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach(), conditions)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            #errD_aux = criterion_classes(classes, conditions)
            #errD_fake = errD_dis + errD_aux
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            # D(G(z)) before D update
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            d_losses += errD.item()
            
            # train G more than D
            for g_iter in range(G_iters):
                '''
                try:
                    images, conditions = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    images, conditions = next(train_iterator)
                '''

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                noise = torch.randn(b_size, 100, 1, 1, device=device)
                fake = netG(noise, conditions)
                label.fill_(real_label) # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake, conditions)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                #errG_aux = criterion_classes(classes, conditions)
                #errG = errG_dis + errG_aux
                # Calculate gradients for G
                errG.backward()
                # D(G(z)) after D update
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()
            g_losses += errG.item()
                
                
            # Output training stats
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
        
        netD.eval()
        netG.eval()
        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = netG(fixed_noise, test_conditions)
            
        score = eval_model.eval(fake, test_conditions)
        
        if score > best_score:
            best_score = score
            fake = fake.detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            torch.save(netG.state_dict(), 'best_model.pt')
    
        # Save Losses for plotting later
        G_losses.append(g_losses)
        D_losses.append(d_losses)
        print('epoch ' + str(epoch) + ' G loss: ' + str(g_losses))
        print('epoch ' + str(epoch) + ' D loss: ' + str(d_losses))
        print('epoch ' + str(epoch) + ' score: ' + str(score))
        print()
        with open('record.txt', 'a') as f:
            f.write('epoch ' + str(epoch) + ' G loss: ' + str(g_losses) + '\n')
            f.write('epoch ' + str(epoch) + ' D loss: ' + str(d_losses) + '\n')
            f.write('epoch ' + str(epoch) + ' score: ' + str(score) + '\n')
            f.write('\n')
    
    
    print('\nbest score: ' + str(best_score))
    
    import matplotlib.pyplot as plt
    from PIL import Image
    import matplotlib.animation as animation
    from IPython.display import HTML
    
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    
    cnt = 1
    for i in img_list:
        ndarr = i.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.save('generated_image/'+str(cnt)+'.png')
        cnt += 1
    
    '''
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    
    HTML(ani.to_jshtml())
    '''

if __name__ == '__main__':
    main()






