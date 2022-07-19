import torch
import os
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from numpy import loadtxt

default_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

class dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):
        assert mode == 'train' or mode == 'test' or mode == 'validate'
        
        self.mode = mode
        if mode == 'train':
            self.data_dir = 'train'
            self.ordered = False
            self.seq_len = args.n_past + args.n_future
        elif mode == 'validate':
            self.data_dir = 'validate'
            self.ordered = True
            #self.seq_len = args.n_eval
            self.seq_len = args.n_past + args.n_future
        else:
            self.data_dir = 'test'
            self.ordered = True
            self.seq_len = args.n_past + args.n_future


        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))  # ex: train/traj_512_to_767.tfrecords/0
        
        self.seed_is_set = False
        self.d = 0
        self.d_con = 0
        self.transform = transform
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            
    def __len__(self):
        return 10000
        
    def get_seq(self):
        if self.ordered:
            d = self.dirs[self.d]
            if self.d == len(self.dirs) - 1:
                self.d = 0
            else:
                self.d += 1
        else:
            d = self.dirs[np.random.randint(len(self.dirs))]

        image_seq = []
        for i in range(self.seq_len):
            fname = '%s/%d.png' % (d, i)
            img = Image.open(fname)
            img = self.transform(img)
            image_seq.append(img)

        self.d_con = d
        return image_seq
    
    def get_csv(self):
        file = open(str(self.d_con) + '/actions.csv', 'r')
        data = loadtxt(file, delimiter=',')

        file2 = open(str(self.d_con) + '/endeffector_positions.csv', 'r')
        data2 = loadtxt(file2, delimiter=',')
        
        condition = []
        for i in range(self.seq_len):
            act_and_pos = np.concatenate((data[i], data2[i]), axis=None)
            act_and_pos = torch.FloatTensor(act_and_pos)
            condition.append(act_and_pos)

        return condition
    
    def __getitem__(self, index):
        self.set_seed(index)
        seq = self.get_seq()
        cond = self.get_csv()
        if self.mode == 'test':
            return seq, cond, self.d_con
        else:
            return seq, cond

def bair_robot_pushing_dataset(args, mode):
    return dataset(args, mode)



