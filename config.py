import os
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.models as models
import torch.backends.cudnn as cudnn
import random
import numpy as np
import logging
from datasets.pedes import CuhkPedes
from models.model import Model
from utils import directory

logger = logging.getLogger()
logger.setLevel(logging.INFO)


numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def data_config(dataset_dir, batch_size, split, max_length, transform):
    data_split = CuhkPedes(dataset_dir, split, max_length, transform)
    if split == 'train':
        shuffle = True
    else:
        shuffle = False
    loader = data.DataLoader(data_split, batch_size, shuffle=shuffle, num_workers=4)
    return loader


def network_config(args, split='train', param=None, resume=False, pretrained_path=None):
    network = Model(args)
    network = nn.DataParallel(network).cuda()
    cudnn.benchmark = True
    args.start_epoch = 0
    if split != 'train':
        args.mobilenet_pretrained = False
    if args.mobilenet_pretrained:
        # pretrained mobilenet
        print('==> Loading from mobilenet pretrained modes')
        network_dict = network.state_dict()
        mobilenet_pretrained = torch.load('mobilenet_sgd_68.848.pth.tar')['state_dict']
        pretrained_dict = {'module.mobilenet_v1.' + k[7:]:v for k,v in mobilenet_pretrained.items()}
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in network_dict}
        network_dict.update(pretrained_dict)
        network.load_state_dict(network_dict)
    # optionally resume from a checkpoint
    elif resume:
        if os.path.isfile(pretrained_path):
            print('==> Loading checkpoint "{}"'.format(pretrained_path))
            checkpoint = torch.load(pretrained_path)
            args.start_epoch = checkpoint['epoch'] + 1
            # best_prec1 = checkpoint['best_prec1']
            network.load_state_dict(checkpoint['state_dict'])
    # optimizer
    if split == 'train':
        mobilenet_params = list(map(id, network.module.mobilenet_v1.parameters()))
        other_params = filter(lambda p: id(p) not in mobilenet_params, network.parameters())
        other_params = list(other_params)
        if param is not None:
            other_params.extend(list(param))
        param_groups = [{'params':other_params},
            {'params':network.module.mobilenet_v1.parameters(), 'weight_decay':args.wd}]
        optimizer = torch.optim.Adam(
            param_groups,
            lr = args.lr, betas=(args.adam_alpha, args.adam_beta), eps=args.epsilon)
        #optimizer = torch.optim.Adam(network.parameters(),args.lr,betas=(args.adam_alpha, args.adam_beta), eps=args.epsilon)
    else:
        optimizer = None
    print('Total params: %2.fM' % (sum(p.numel() for p in network.parameters()) / 1000000.0))
    # seed
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    return network, optimizer


def log_config(args, ca):
    logging.info(args)
    filename = args.log_dir +'/' + ca + '.log'
    handler = logging.FileHandler(filename)                                                                                   
    handler.setLevel(logging.INFO)                                                                                                      
    formatter = logging.Formatter('%(message)s')                                                                                           
    handler.setFormatter(formatter)                                                                                                     
    logger.addHandler(handler) 


def dir_config(args):
    if not os.path.exists(args.dataset_dir):
        raise ValueError('Supply the dataset directory with --dataset_dir')
    directory.makedir(args.log_dir)
    # save checkpoint
    directory.makedir(args.checkpoint_dir)
    directory.makedir(os.path.join(args.checkpoint_dir,'model_best'))
