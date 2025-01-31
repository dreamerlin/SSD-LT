#!/usr/bin/env python
import argparse
import builtins
import os
import random
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from datasets import build_dataset

from tqdm import tqdm

import models.resnet
import SSD_LT.loader
import SSD_LT.builder

from SSD_LT.utils import *
import utils

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision
from mcloader.transforms_ss import *

CLIP_DEFAULT_MEAN = (0.4815, 0.4578, 0.4082)
CLIP_DEFAULT_STD = (0.2686, 0.2613, 0.2758)


parser = argparse.ArgumentParser(description='SSD_LT ImageNet-LT Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=135, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=115712, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)')

parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--output_dir', type=str,
                    default='weights/stage_i/',
                    help='path to store checkpoints')
parser.add_argument('--num-segments', type=int, default=8)
parser.add_argument('--dense-sample', action='store_true')
parser.add_argument('--test-crops', type=int, default=1)
parser.add_argument('--num-clips', type=int, default=10)
parser.add_argument('--test-batch-size', type=int, default=0)
parser.add_argument('--use-softmax', action='store_true', default=False)
parser.add_argument('--twice-sample', action='store_true')
parser.add_argument('--dataset', type=str, default='Kinetics')
parser.add_argument('--index-bias', type=int, default=0)
parser.add_argument('--context-length', type=int, default=77)
parser.add_argument('--io-backend', type=str, default='petrel')


best_acc1 = 0

def build_transform_val(is_train, args, split):
    DEFAULT_MEAN = CLIP_DEFAULT_MEAN if args.clip_ms else IMAGENET_DEFAULT_MEAN
    DEFAULT_STD = CLIP_DEFAULT_STD if args.clip_ms else IMAGENET_DEFAULT_STD
    scale_size = args.input_size * 256 // 224
    if is_train:
        unique = torchvision.transforms.Compose([GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66]),
                                                 GroupRandomHorizontalFlip(is_sth='some' in args.data_set),
                                                 GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4,
                                                                        saturation=0.2, hue=0.1),
                                                 GroupRandomGrayscale(p=0.2),
                                                 GroupGaussianBlur(p=0.0),
                                                 GroupSolarization(p=0.0)])
    else:
        if args.test_crops == 1 or split == 'val':
            unique = torchvision.transforms.Compose([GroupScale(scale_size),
                                                     GroupCenterCrop(args.input_size)])
        elif args.test_crops == 3:
            unique = torchvision.transforms.Compose([GroupFullResSample(args.input_size, scale_size, flip=False)])
        elif args.test_crops == 5:
            unique = torchvision.transforms.Compose([GroupOverSample(args.input_size, scale_size, flip=False)])
        elif args.text_crops == 10:
            unique = torchvision.transforms.Compose([GroupOverSample(args.input_size, scale_size)])
        else:
            raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(args.test_crops))

    common = torchvision.transforms.Compose([Stack(roll=False),
                                             ToTorchFormatTensor(div=True),
                                             GroupNormalize(DEFAULT_MEAN,
                                                            DEFAULT_STD)])
    return torchvision.transforms.Compose([unique, common])

def main():
    args = parser.parse_args()
    args = utils.update_from_config(args)
    args.num_class = 400
    args.clip_ms = True
    args.input_size = 224
    scale_size = args.input_size * 256 // 224
    args.scale_size = scale_size
    args.only_video = True
    args.select_num = 50

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model")
    model = SSD_LT.builder.SSFL(
        models.resnet.resnext50_32x4d,
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.num_class,
        args.num_segments)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    DEFAULT_MEAN = CLIP_DEFAULT_MEAN if args.clip_ms else IMAGENET_DEFAULT_MEAN
    DEFAULT_STD = CLIP_DEFAULT_STD if args.clip_ms else IMAGENET_DEFAULT_STD

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    augmentation = [
        GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66]),
        GroupRandomHorizontalFlip(is_sth='some' in args.data_set),
        GroupRandomColorJitter(p=1.0, brightness=0.4, contrast=0.4,
                               saturation=0.2, hue=0.1),
        # GroupRandomGrayscale(p=0.2),
        # GroupGaussianBlur(p=0.0),
        # GroupSolarization(p=0.0),
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize(DEFAULT_MEAN, DEFAULT_STD)]

    k_augmentation = [
        GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66]),
        GroupRandomHorizontalFlip(is_sth='some' in args.data_set),
        GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4,
                               saturation=0.2, hue=0.1),
        GroupRandomGrayscale(p=0.2),
        GroupGaussianBlur(p=0.5),
        # GroupSolarization(p=0.0),
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize(DEFAULT_MEAN, DEFAULT_STD)
    ]

    train_aug = SSD_LT.loader.TwoCropsTransform(transforms.Compose(augmentation),
                                                transforms.Compose(k_augmentation))
    val_aug = build_transform_val(is_train=False, args=args, split='val')
    test_aug = build_transform_val(is_train=False, args=args, split='test')


    train_dataset, args.nb_classes = build_dataset(split="train", args=args, transform=train_aug)
    if args.test:
        test_dataset, _ = build_dataset(split="test", args=args, transform=val_aug)
    val_dataset, _ = build_dataset(split="val", args=args, transform=test_aug)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False))

    if args.evaluate:
        evaluate(test_loader, model, train_dataset.targets, phase='Test', args=args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        top1 = evaluate(val_loader, model, train_dataset.targets, args=args)

        is_best = top1 > best_acc1
        best_acc1 = max(top1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_acc1': best_acc1,
            }, is_best=is_best, filename='{}/last_checkpoint.pth.tar'.format(args.output_dir))
    
    print('Evaluate test set with last checkpoint...')
    evaluate(test_loader, model, train_dataset.targets, phase='Test', args=args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.3f')
    cls_losses = AverageMeter('Cls Losses', ':6.3f')
    cont_losses = AverageMeter('Cont Losses', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, cls_losses, cont_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, label, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)
            index = index.cuda(args.gpu, non_blocking=True)

        # compute output
        pred, cont_loss = model(im_q=images[0], im_k=images[1], labels=label)
        cls_loss = criterion(pred, label)
        loss = cls_loss + cont_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(pred, label, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        cls_losses.update(cls_loss.item(), images[0].size(0))
        cont_losses.update(cont_loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def evaluate(dataloader, model, train_labels, phase='Validation', args=None):
    model.eval()
    all_logits = torch.empty((0, args.num_class)).cuda(args.gpu)
    all_labels = torch.empty(0, dtype=torch.long).cuda(args.gpu)

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc=phase, disable=args.gpu!=0):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)
            logits = model(images)
            logits_gather = SSD_LT.builder.concat_all_gather(logits)
            labels_gather = SSD_LT.builder.concat_all_gather(labels)
            all_logits = torch.cat((all_logits, logits_gather))
            all_labels = torch.cat((all_labels, labels_gather))
        _, preds = all_logits.max(dim=1)
        
    many_acc_top1, \
    median_acc_top1, \
    low_acc_top1 = shot_acc(preds, all_labels, train_labels)

    overall_acc_top1 = mic_acc_cal(preds, all_labels)
    print_str = ['\n\n',
                    'Phase: %s' 
                    % (phase),
                    '\n\n',
                    'Evaluation_overall_accuracy_top1: %.3f' 
                    % (overall_acc_top1),
                    '\n',
                    'Many_shot_accuracy_top1: %.3f' 
                    % (many_acc_top1),
                    'Median_shot_accuracy_top1: %.3f' 
                    % (median_acc_top1),
                    'Low_shot_accuracy_top1: %.3f' 
                    % (low_acc_top1),
                    '\n']
    print(*print_str)
    return overall_acc_top1


if __name__ == '__main__':
    main()
