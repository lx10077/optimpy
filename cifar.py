from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.classification as models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, get_flat_grad_from, get_flat_para_from
from optimizers.riemann import RiemannSGD

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Data sets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate for x')
parser.add_argument('--lr_r', '--learning-rate-r', default=0.1, type=float,
                    metavar='LR', help='initial learning rate for r')
parser.add_argument('--lr_w', '--learning-rate-w', default=0.1, type=float,
                    metavar='LR', help='initial learning rate for w')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Check points
parser.add_argument('-c', '--save_checkpoint', default='save_checkpoint', type=str, metavar='PATH',
                    help='path to save save_checkpoint (default: save_checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest save_checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='ResNet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
log_name = "resnet18-rSGD"


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last save_checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    try:
        model = models.__dict__[args.arch](num_classes=num_classes)
    except Exception as e:
        print("Fail to locate the model.")
        print("All possible models:", model_names)
        raise Exception(e)

    try:
        print('Using', torch.cuda.device_count(), 'GPUs.')
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        print('Using CUDA.')
    except Exception as e:
        print("Fail to use DataParallel.")
        print(Exception(e))
        pass

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = RiemannSGD(model.parameters(), lr_r=args.lr_r, lr_w=args.lr_w,
                           momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'classification-10-' + args.arch
    if args.resume:
        # Load save_checkpoint.
        print('==> Resuming from save_checkpoint..')
        if type(args.resume, str):
            assert os.path.isfile(args.resume), 'Error: no save_checkpoint directory found!'
        if type(args.resume, bool):
            print("Using default save_checkpoint..")
            args.resume = "save_checkpoint/" + log_name + '-save_checkpoint.pth'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, log_name + '.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, log_name + '.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
    writer = SummaryWriter(os.path.join(args.checkpoint, log_name))

    if args.evaluate:
        print('\nEvaluation only')
        test_writer = SummaryWriter(os.path.join(args.checkpoint, log_name + '-test'))
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda, test_writer)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda, writer)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda, writer)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


def train(trainloader, model, criterion, optimizer, epoch, use_cuda, writer):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    grad_norms = AverageMeter()
    para_norms = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # measure gradient norms and parameter norms
        grad_norm = torch.norm(get_flat_grad_from(model.parameters()))
        para_norm = torch.norm(get_flat_para_from(model.parameters()))
        grad_norms.update(grad_norm.data[0], inputs.size(0))
        para_norms.update(para_norm.data[0], inputs.size(0))

        i = epoch * len(trainloader) + batch_idx
        writer.add_scalar("batch_loss", loss.item(), i)
        writer.add_scalar("batch_loss_avg", losses.avg, i)
        writer.add_scalar("batch_top1", top1.avg, i)
        writer.add_scalar("batch_top5", top5.avg, i)
        writer.add_scalar("batch_grad_norm", grad_norm.data[0], i)
        writer.add_scalar("batch_grad_norm_avg", grad_norms.avg, i)
        writer.add_scalar("batch_para_norm", para_norm.data[0], i)
        writer.add_scalar("batch_para_norm_avg", para_norms.avg, i)

        # plot progress
        msg = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} ' \
              '| ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}' \
              ''.format(batch=batch_idx + 1, size=len(trainloader), data=data_time.avg,
                        bt=batch_time.avg, total=bar.elapsed_td, eta=bar.eta_td, loss=losses.avg,
                        top1=top1.avg, top5=top5.avg)
        print(msg)
        bar.suffix = msg
        bar.next()
    bar.finish()
    return losses.avg, top1.avg


def test(testloader, model, criterion, epoch, use_cuda, writer):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        i = epoch * len(testloader) + batch_idx
        writer.add_scalar("test_batch_loss", loss.item(), i)
        writer.add_scalar("test_batch_loss_avg", losses.avg, i)
        writer.add_scalar("test_batch_top1", top1.avg, i)
        writer.add_scalar("test_batch_top5", top5.avg, i)

        # plot progress
        msg = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} ' \
              '| ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}' \
              ''.format(batch=batch_idx + 1, size=len(testloader), data=data_time.avg,
                        bt=batch_time.avg, total=bar.elapsed_td, eta=bar.eta_td, loss=losses.avg,
                        top1=top1.avg, top5=top5.avg)
        bar.suffix = msg
        bar.next()
    bar.finish()
    return losses.avg, top1.avg


def save_checkpoint(states, is_best, checkpoint='save_checkpoint', suffix='-save_checkpoint.pth'):
    filename = log_name + suffix
    filepath = os.path.join(checkpoint, filename)
    torch.save(states, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, log_name + '-model_best.pth'))


def load_checkpoint(checkpoint='save_checkpoint', suffix='-save_checkpoint.pth'):
    filename = log_name + suffix
    filepath = os.path.join(checkpoint, filename)
    return torch.load(filepath)


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        try:
            state['lr_r'] *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr_r'] = state['lr_r']
            state['lr_w'] *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr_w'] = state['lr_w']
        except:
            state['lr'] *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
