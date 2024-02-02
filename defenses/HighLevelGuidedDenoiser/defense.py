"""Sample Pytorch defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import math
import numpy as np

import torch
import torch.autograd as autograd
import torch.utils.data as data
import torchvision
import torchvision.datasets.folder
import torchvision.transforms as transforms

from dataset import Dataset
from res152_wide import get_model as get_model1
from inres import get_model as  get_model2
from v3 import get_model as get_model3
from resnext101 import get_model as get_model4

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR', default='',
                    help='Input directory with images.')
parser.add_argument('--output_file', metavar='FILE', default='',
                    help='Output file to save labels.')
parser.add_argument('--checkpoint_path', default=None,
                    help='Path to network checkpoint.')
parser.add_argument('--checkpoint_path2', default=None,
                    help='Path to network checkpoint.')
parser.add_argument('--img-size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='Batch size (default: 32)')
parser.add_argument('--no-gpu', action='store_true', default=False,
                    help='disables GPU training')


class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """

    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor
    
    

def main():
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print("Error: Invalid input folder %s" % args.input_dir)
        exit(-1)
    if not args.output_file:
        print("Error: Please specify an output file")
        exit(-1)
        
    tf = transforms.Compose([
           transforms.Scale([299,299]),
            transforms.ToTensor()
    ])

    mean_torch = autograd.Variable(torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([1,3,1,1]).astype('float32')).cuda(), volatile=True)
    std_torch = autograd.Variable(torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape([1,3,1,1]).astype('float32')).cuda(), volatile=True)
    mean_tf = autograd.Variable(torch.from_numpy(np.array([0.5, 0.5, 0.5]).reshape([1,3,1,1]).astype('float32')).cuda(), volatile=True)
    std_tf = autograd.Variable(torch.from_numpy(np.array([0.5, 0.5, 0.5]).reshape([1,3,1,1]).astype('float32')).cuda(), volatile=True)
    

    dataset = Dataset(args.input_dir, transform=tf)
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    config, resmodel = get_model1()
    config, inresmodel = get_model2()
    config, incepv3model = get_model3()
    config, rexmodel = get_model4()
    net1 = resmodel.net    
    net2 = inresmodel.net
    net3 = incepv3model.net
    net4 = rexmodel.net

    checkpoint = torch.load('denoise_res_015.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        resmodel.load_state_dict(checkpoint['state_dict'])
    else:
        resmodel.load_state_dict(checkpoint)

    checkpoint = torch.load('denoise_inres_014.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        inresmodel.load_state_dict(checkpoint['state_dict'])
    else:
        inresmodel.load_state_dict(checkpoint)

    checkpoint = torch.load('denoise_incepv3_012.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        incepv3model.load_state_dict(checkpoint['state_dict'])
    else:
        incepv3model.load_state_dict(checkpoint)
    
    checkpoint = torch.load('denoise_rex_001.ckpt')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        rexmodel.load_state_dict(checkpoint['state_dict'])
    else:
        rexmodel.load_state_dict(checkpoint)

    if not args.no_gpu:
        inresmodel = inresmodel.cuda()
        resmodel = resmodel.cuda()
        incepv3model = incepv3model.cuda()
        rexmodel = rexmodel.cuda()
    inresmodel.eval()
    resmodel.eval()
    incepv3model.eval()
    rexmodel.eval()

    outputs = []
    for batch_idx, (input, _) in enumerate(loader):
        if not args.no_gpu:
            input = input.cuda()
        input_var = autograd.Variable(input, volatile=True)
        input_tf = (input_var-mean_tf)/std_tf
        input_torch = (input_var - mean_torch)/std_torch

        #clean1 = net1.denoise[0](input_torch)        
        #clean2 = net2.denoise[0](input_tf)
        #clean3 = net3.denoise(input_tf)

        #labels1 = net1(clean1,False)[-1]
        #labels2 = net2(clean2,False)[-1]
        #labels3 = net3(clean3,False)[-1]
        
        labels1 = net1(input_torch,True)[-1]
        labels2 = net2(input_tf,True)[-1]
        labels3 = net3(input_tf,True)[-1]
        labels4 = net4(input_torch,True)[-1]

        labels = (labels1+labels2+labels3+labels4).max(1)[1] + 1  # argmax + offset to match Google's Tensorflow + Inception 1001 class ids
        outputs.append(labels.data.cpu().numpy())
    outputs = np.concatenate(outputs, axis=0)

    with open(args.output_file, 'w') as out_file:
        filenames = dataset.filenames()
        for filename, label in zip(filenames, outputs):
            filename = os.path.basename(filename)
            out_file.write('{0},{1}\n'.format(filename, label))

if __name__ == '__main__':
    main()
