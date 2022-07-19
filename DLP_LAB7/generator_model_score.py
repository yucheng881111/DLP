# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 00:08:20 2022

@author: user
"""

import torch
import json
from models import Generator
from evaluator import evaluation_model
import torchvision.utils as vutils
from PIL import Image

def get_test_conditions():
    with open('objects.json', 'r') as file:
        classes = json.load(file)
    with open('test.json', 'r') as file:    # new_test.json
        test_conditions_list = json.load(file)

    labels = torch.zeros(len(test_conditions_list), len(classes))
    for i in range(len(test_conditions_list)):
        for condition in test_conditions_list[i]:
            labels[i, int(classes[condition])] = 1.

    return labels


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Generator().to(device)
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    eval_model = evaluation_model()
    test_conditions = get_test_conditions().to(device)
    
    seed = 10
    torch.cuda.manual_seed(seed)
    
    li = []
    for i in range(10):
        with torch.no_grad():
            noise = torch.randn(len(test_conditions), 100, 1, 1, device=device)
            fake = model(noise, test_conditions)
            
            img = vutils.make_grid(fake, padding=2, normalize=True)
            ndarr = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save('generated_image_best/' + str(i+1) + '.png')
            
        score = eval_model.eval(fake, test_conditions)
        li.append(score)
        print('score: ' + str(score))
    
    print()
    print('max score: ' + str(max(li)))
    print('avg score: ' + str(sum(li) / 10))
    


if __name__ == '__main__':
    main()

