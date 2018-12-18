import os
import sys
import shutil
import time
import logging
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.transforms as transforms
from utils.metric import AverageMeter, Loss, constraints_loss
from test import test
from config import data_config, network_config, adjust_lr
from train_config import config

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def save_checkpoint(state, epoch, dst, is_best):
    filename = os.path.join(dst, str(args.start_epoch + epoch)) + '.pth.tar'
    torch.save(state, filename)
    if is_best:
        dst_best = os.path.join(dst, 'model_best', str(epoch)) + '.pth.tar'
        shutil.copyfile(filename, dst_best)


def train(epoch, train_loader, network, optimizer, compute_loss, args):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    image_pre = AverageMeter()
    text_pre = AverageMeter()

    # switch to train mode
    network.train()

    end = time.time()
    for step, (images, captions, labels, captions_length) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        captions = captions.cuda()

        # compute loss
        image_embeddings, text_embeddings = network(images, captions, captions_length)
        cmpm_loss, cmpc_loss, loss, image_precision, text_precision, pos_avg_sim, neg_arg_sim = compute_loss(image_embeddings, text_embeddings, labels)
        

        if step % 10 == 0:
            print('epoch:{}, step:{}, cmpm_loss:{:.3f}, cmpc_loss:{:.3f}'.format(epoch, step, cmpm_loss, cmpc_loss))

        # constrain embedding with the same id at the end of one epoch
        if (args.constraints_images or args.constraints_text) and step == len(train_loader) - 1:
            con_images, con_text = constraints_loss(train_loader, network, args)
            loss += (con_images + con_text)
            print('epoch:{}, step:{}, con_images:{:.3f}, con_text:{:.3f}'.format(epoch, step, con_images, con_text))
        

        # compute gradient and do ADAM step
        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm(network.parameters(), 5)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        train_loss.update(loss, images.shape[0])
        image_pre.update(image_precision, images.shape[0])
        text_pre.update(text_precision, images.shape[0])
        
        
        
    return train_loss.avg, batch_time.avg, image_pre.avg, text_pre.avg


def main(args):
    # transform
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # data
    train_loader = data_config(args.dataset_dir, args.batch_size, 'train', args.max_length, train_transform)
    #val_loader = data_config(args.dataset_dir, 64, 'val', args.max_length, val_transform)
    
    # loss
    compute_loss = Loss(args)
    nn.DataParallel(compute_loss).cuda()
    
    # network
    network, optimizer = network_config(args, 'train', compute_loss.parameters(), args.resume, args.pretrained_path)
    #ac_i2t_best = 0.0
    #ac_t2i_best = 0.0
    for epoch in range(args.num_epoches):
        # train for one epoch
        train_loss, train_time, image_precision, text_precision = train(epoch, train_loader, network, optimizer, compute_loss, args)
        # evaluate on validation set
        is_best = False
        print('Train done for epoch-{}'.format(epoch))
        '''
        ac_top1_t2i, ac_top1_i2t, ac_top10_t2i, ac_top10_i2t, val_time = test(val_loader, network, args)
        # remember best top@1 and top@10, save checkpoint and logging to the console
        is_best = False
        if ac_top1_i2t > ac_i2t_best:
            ac_i2t_best = ac_top1_i2t
            is_best = True
        if ac_top1_t2i > ac_t2i_best:
            ac_t2i_best = ac_top1_t2i
            is_best = True
        '''
        state = {'state_dict': network.state_dict(), 'W': compute_loss.W, 'epoch': args.start_epoch + epoch}
        #         'ac': [ac_top1_i2t, ac_top10_i2t, ac_top1_t2i, ac_top10_t2i],
        #         'best_ac': [ac_i2t_best, ac_t2i_best]}
        save_checkpoint(state, epoch, args.checkpoint_dir, is_best)
        logging.info('Epoch:  [{}|{}], train_time: {:.3f}, train_loss: {:.3f}'.format(epoch, args.num_epoches, train_time, train_loss))
        logging.info('image_precision: {:.3f}, text_precision: {:.3f}'.format(image_precision, text_precision))
        adjust_lr(optimizer, epoch, args)
        #logging.info('val_time: {:.3f}, top1_i2t: {:.3f}, top10_i2t: {:.3f}, top1_t2i: {:.3f}, top10_t2i: {:.3f}'.format(val_time, ac_top1_i2t, ac_top10_i2t, ac_top1_t2i, ac_top10_t2i))
    #print('best i2t top1 ac:%.3f' % ac_i2t_best)
    #print('best t2i top1 ac:%.3f' % ac_t2i_best)


if __name__ == "__main__":
    args = config()
    main(args)
