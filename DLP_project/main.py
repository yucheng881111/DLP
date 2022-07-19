# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 21:46:17 2022

@author: user
"""

import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import SimulaterDataset
from torch.utils.data import DataLoader
from model import six_layer_CNN
from tqdm import tqdm

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size = 64
    DataSet = SimulaterDataset('train')
    train_loader =  DataLoader(DataSet, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    DataSet_test = SimulaterDataset('test')
    test_loader = DataLoader(DataSet_test, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    model = six_layer_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    #from adam_lrd import Adam_LRD
    #optimizer = Adam_LRD(model.parameters(), lr=lr, betas=(0.5, 0.999), dropout=0.5)
    
    best_acc = 0
    epochs = 50
    progress = tqdm(total=epochs)
    for epoch in range(epochs):
        Loss = 0
        total_train = 0
        correct_train = 0
        model.train()
        for _, (data, label, id) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device, dtype = torch.float)
            label = label.to(device, dtype = torch.long)
            data = data.view(-1, 1, 750)
            #print(data.size())  # 64 * 1 * 750
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            Loss += loss.item()
            # Get predictions from the maximum value
            predicted = torch.max(output.data, 1)[1]
            # Total number of labels
            total_train += len(label)
            # Total correct predictions
            correct_train += (predicted == label).float().sum()
        
        train_accuracy = 100 * (correct_train / total_train)
        print('epoch ' + str(epoch))
        print('loss: ' + str(Loss) + ' acc: ' + str(train_accuracy.item()))
        
        with open('record.txt', 'a') as f:
            f.write('epoch ' + str(epoch) + '\n')
            f.write('loss: ' + str(Loss) + ' acc: ' + str(train_accuracy.item()) + '\n')

    
        total_test = 0
        correct_test = 0
        model.eval()
        with torch.no_grad():
            for _, (data, label, id) in enumerate(test_loader):
                data = data.to(device, dtype = torch.float)
                label = label.to(device, dtype = torch.long)
                data = data.view(-1, 1, 750)
                output = model(data)
                # Get predictions from the maximum value
                predicted = torch.max(output.data, 1)[1]
                # Total number of labels
                total_test += len(label)
                # Total correct predictions
                correct_test += (predicted == label).float().sum()
       
        test_accuracy = 100 * (correct_test / total_test)
        print('\ntest acc: ' + str(test_accuracy.item()))
        print()

        with open('record.txt', 'a') as f:
            f.write('test acc: ' + str(test_accuracy.item()) + '\n\n')

        progress.update(1)

        if test_accuracy > best_acc:
            best_acc = test_accuracy
            # save model
            torch.save(model.state_dict(), 'best_model.pt')
            
    print('\nbest test acc: ' + str(best_acc.item()))
        


if __name__ == '__main__':
    main()





