import os
import sys
import time
import shutil
import logging
import gc
import torch
import torchvision.transforms as transforms
from utils.metric import AverageMeter, compute_topk
from test_config import config
from config import data_config, network_config


def test(data_loader, network, args):
    batch_time = AverageMeter()

    # switch to evaluate mode
    network.eval()
    max_size = 64 * len(data_loader)
    images_bank = torch.zeros((max_size, args.feature_size)).cuda()
    text_bank = torch.zeros((max_size,args.feature_size)).cuda()
    labels_bank = torch.zeros(max_size).cuda()
    index = 0
    with torch.no_grad():
        end = time.time()
        for images, captions, labels, captions_length in data_loader:
            images = images.cuda()
            captions = captions.cuda()

            interval = images.shape[0]
            image_embeddings, text_embeddings = network(images, captions, captions_length)
            images_bank[index: index + interval] = image_embeddings
            text_bank[index: index + interval] = text_embeddings
            labels_bank[index:index + interval] = labels
            batch_time.update(time.time() - end)
            end = time.time()
            
            index = index + interval
        
        images_bank = images_bank[:index]
        text_bank = text_bank[:index]
        labels_bank = labels_bank[:index]
        #[ac_top1_t2i, ac_top10_t2i] = compute_topk(text_bank, images_bank, labels_bank, labels_bank, [1,10])
        #[ac_top1_i2t, ac_top10_i2t] = compute_topk(images_bank, text_bank, labels_bank, labels_bank, [1,10])
        ac_top1_i2t, ac_top10_i2t, ac_top1_t2i, ac_top10_t2i = compute_topk(images_bank, text_bank, labels_bank, labels_bank, [1,10], True)
        return ac_top1_i2t, ac_top10_i2t, ac_top1_t2i, ac_top10_t2i, batch_time.avg


def main(args):
    # need to clear the pipeline
    # top1 & top10 need to be chosen in the same params ???
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_loader = data_config(args.image_dir, args.anno_dir, 64, 'test', args.max_length, test_transform)

    ac_i2t_top1_best = 0.0
    ac_i2t_top10_best = 0.0
    ac_t2i_top1_best = 0.0
    ac_t2i_top10_best = 0.0
    i2t_models = os.listdir(args.model_path)
    i2t_models.sort()
    for i2t_model in i2t_models:
        model_file = os.path.join(args.model_path, i2t_model)
        if os.path.isdir(model_file):
            continue
        epoch = i2t_model.split('.')[0]
        if int(epoch) >= args.epoch_ema:
            ema = True
        else:
            ema = False
        network, _ = network_config(args, [0], 'test', None, True, model_file, ema)
        ac_top1_i2t, ac_top10_i2t, ac_top1_t2i, ac_top10_t2i, test_time = test(test_loader, network, args)
        if ac_top1_t2i > ac_t2i_top1_best:
            ac_i2t_top1_best = ac_top1_i2t
            ac_i2t_top10_best = ac_top10_i2t
            ac_t2i_top1_best = ac_top1_t2i
            ac_t2i_top10_best = ac_top10_t2i
            dst_best = os.path.join(args.model_path, 'model_best', str(epoch)) + '.pth.tar'
            shutil.copyfile(model_file, dst_best)
         
        logging.info('epoch:{}'.format(epoch))
        logging.info('top1_t2i: {:.3f}, top10_t2i: {:.3f}, top1_i2t: {:.3f}, top10_i2t: {:.3f}'.format(
            ac_top1_t2i, ac_top10_t2i, ac_top1_i2t, ac_top10_i2t))
    logging.info('t2i_top1_best: {:.3f}, t2i_top10_best: {:.3f}, i2t_top1_best: {:.3f}, i2t_top10_best: {:.3f}'.format(
            ac_t2i_top1_best, ac_t2i_top10_best, ac_i2t_top1_best, ac_i2t_top10_best))
    logging.info(args.model_path)
    logging.info(args.log_dir)

if __name__ == '__main__':
    args = config()
    main(args)
