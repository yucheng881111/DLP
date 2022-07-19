import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
#from utils import init_weights, kl_criterion, plot_pred, plot_rec, finn_eval_seq
from utils import init_weights, kl_criterion, finn_eval_seq

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--log_dir', default='./logs/fp', help='base directory to save logs')
    parser.add_argument('--model_dir', default='model.pth', help='base directory to save logs')
    parser.add_argument('--data_root', default='./data/processed_data', help='root directory for data')
    parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=0, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=2, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3, help='The number of cycle for kl annealing (if use cyclical mode)')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict at eval time')
    parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
    parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading threads')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
    parser.add_argument('--cuda', default=True, action='store_true')

    args = parser.parse_args()
    return args


def pred(validate_seq, validate_cond, modules, args, device):
    # initialize the hidden state.
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    modules['prior'].hidden = modules['prior'].init_hidden()

    pred_seq = []
    for i in range(args.n_past):
        pred_seq.append(validate_seq[i])

    
    # calculate full given sequence
    h_seq = [modules['encoder'](validate_seq[i]) for i in range(args.n_past)]
    h, skip = h_seq[-1]

    for i in range(args.n_future):
        # h_t
        h = h_seq[-1][0]
        #z_t = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_()
        z_t, _, _ = modules['prior'](h)
        h_pred = modules['frame_predictor'](torch.cat([h, z_t], 1))
        x_pred = modules['decoder'](h_pred, skip, validate_cond[args.n_past - 1 + i])
        h_seq.append(modules['encoder'](x_pred))
        pred_seq.append(x_pred)

    return pred_seq


def main():
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'
    
    assert args.n_past + args.n_future <= 30 and args.n_eval <= 30
    assert 0 <= args.tfr and args.tfr <= 1
    assert 0 <= args.tfr_start_decay_epoch 
    assert 0 <= args.tfr_decay_step and args.tfr_decay_step <= 1

    
    # load model and continue training from checkpoint
    saved_model = torch.load('best_model/model.pth')
    optimizer = args.optimizer
    model_dir = args.model_dir
    niter = args.niter
    args = saved_model['args']
    args.optimizer = optimizer
    args.model_dir = model_dir
    args.log_dir = '%s/continued' % args.log_dir
    start_epoch = saved_model['last_epoch']

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(args)
    # ------------ build the models  --------------

    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
    prior = saved_model['prior']
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
    
    
    # --------- transfer to device ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    prior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    test_data = bair_robot_pushing_dataset(args, 'test')

    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True)

    test_iterator = iter(test_loader)

    # ---------------- optimizers ----------------
    if args.optimizer == 'adam':
        args.optimizer = optim.Adam
    elif args.optimizer == 'rmsprop':
        args.optimizer = optim.RMSprop
    elif args.optimizer == 'sgd':
        args.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % args.optimizer)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'prior': prior,
        'encoder': encoder,
        'decoder': decoder,
    }

    frame_predictor.eval()
    encoder.eval()
    decoder.eval()
    posterior.eval()
    prior.eval()

    psnr_list = []

    for data_idx in range(88):
        try:
            test_seq, test_cond, idx = next(test_iterator)
        except StopIteration:
            test_iterator = iter(test_loader)
            test_seq, test_cond, idx = next(test_iterator)

        test_seq = [x.to(device) for x in test_seq]
        test_cond = [x.to(device) for x in test_cond]

        with open('generation_record.txt', 'a') as f:
            f.write('idx ' + str(data_idx) + ':')
            f.write('\n')

        pred_seq = pred(test_seq, test_cond, modules, args, device)
        _, _, psnr = finn_eval_seq(test_seq[args.n_past:args.n_past+args.n_future], pred_seq[args.n_past:])

        for i in range(args.batch_size):
            with open('generation_record.txt', 'a') as f:
                f.write('data route: ' + idx[i])
                f.write('\n')
                f.write('seq psnr: ' + str(psnr[i]))
                f.write('\n')
                f.write('psnr: ' + str(np.mean(psnr[i])))
                f.write('\n\n')

        with open('generation_record.txt', 'a') as f:
            f.write('\n')

        psnr_list.append(np.mean(np.concatenate(psnr)))

    ave_psnr = np.mean(np.array(psnr_list))
    

    print(' avg: ' + str(ave_psnr))

    '''
    # save image
    # every seq
    for i in range(12):
        # every batch
        for j in range(12):
            test_img = test_seq[i][j]
            test_array = np.moveaxis(test_img.cpu().detach().numpy()*255, 0, -1)
            test_res_img = Image.fromarray(test_array.astype(np.uint8))
            test_res_img.save('gen_truth/' + str(data_idx) + '/seq_' + str(i) + '_batch_' + str(j) + '.png')

            gen_img = pred_seq[i][j]
            gen_array = np.moveaxis(gen_img.cpu().detach().numpy()*255, 0, -1)
            gen_res_img = Image.fromarray(gen_array.astype(np.uint8))
            gen_res_img.save('gen_pred/' + str(data_idx) + '/seq_' + str(i) + '_batch_' + str(j) + '.png')
    '''
        
if __name__ == '__main__':
    main()
        
