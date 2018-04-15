import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import torch
import time
import shutil
import models.cifar as models
import torch.nn as nn
import torch.backends.cudnn as cudnn
from oputils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


class CifarTrainer(object):
    def __init__(self, dataset, cfg):
        self.best_acc = 0
        assert dataset == 'cifar10' or dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'
        self.dataset = dataset
        self.cfg = cfg

        # Data
        self.train_batch = cfg["train_batch"]
        self.test_batch = cfg["test_batch"]
        self.workers = cfg["workers"]

        # Architecture
        self.arch = cfg["arch"]
        self.drop = cfg["drop"]
        self.depth = cfg["depth"]
        self.cardinality = cfg["cardinality"]
        self.widen_factor = cfg["widen_factor"]
        self.growth_rate = cfg["growth_rate"]
        self.compression_rate = cfg["compression_rate"]

        # Optimization
        self.start_epoch = cfg["start_epoch"]
        self.epochs = cfg["epochs"]
        self.start_epoch = cfg["start_epoch"]
        self.train_batch = cfg["train_batch"]
        self.test_batch = cfg["test_batch"]
        self.gamma = cfg["gamma"]
        self.schedule = cfg["schedule"]
        self.use_cuda = torch.cuda.is_available() & cfg["gpu"]

    def start(self, optimizers, resume=False):
        criterion = nn.CrossEntropyLoss()
        for optim_name, optim in optimizers.items():
            model = self._prepare_model(self.arch)
            checkpoint = None
            if not os.path.isdir(checkpoint):
                mkdir_p(checkpoint)
            self.task(optim_name, model, criterion, optim, self.use_cuda, checkpoint, resume)

    def task(self, name, model, criterion, optimizer, use_cuda, checkpoint, resume):
        trainloader, testloader = self._prepare_data()
        best_acc = 0.
        logger = self._prepare_log(checkpoint, name, resume)

        # Train and val
        for epoch in range(self.start_epoch, self.epochs):
            self.adjust_learning_rate(optimizer, epoch)

            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, self.epochs, state['lr']))

            train_loss, train_acc = self.train(trainloader, model, criterion, optimizer, epoch, use_cuda)
            test_loss, test_acc = self.test(testloader, model, criterion, epoch, use_cuda)

            # append logger file
            logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

            # save model
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=checkpoint)

        logger.close()
        logger.plot()
        savefig(os.path.join(checkpoint, 'log.eps'))

        print('Best acc:')
        print(best_acc)

    def _prepare_data(self):
        print('==> Preparing dataset %s' % self.dataset)
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
        if self.dataset == 'cifar10':
            dataloader = datasets.CIFAR10
            self.num_classes = 10
        else:
            dataloader = datasets.CIFAR100
            self.num_classes = 100

        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=self.train_batch, shuffle=True, num_workers=self.workers)

        testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=self.test_batch, shuffle=False, num_workers=self.workers)
        return trainloader, testloader

    def _prepare_model(self, arch):
        print("==> creating model '{}'".format(arch))

        if arch.startswith('resnext'):
            model = models.__dict__[arch](
                cardinality=self.cardinality,
                num_classes=self.num_classes,
                depth=self.depth,
                widen_factor=self.widen_factor,
                dropRate=self.drop,
            )
        elif arch.startswith('densenet'):
            model = models.__dict__[arch](
                num_classes=self.num_classes,
                depth=self.depth,
                growthrate=self.growth_rate,
                compressionrate=self.compression_rate,
                dropRate=self.drop,
            )
        elif arch.startswith('wrn'):
            model = models.__dict__[arch](
                num_classes=self.num_classes,
                depth=self.depth,
                widen_factor=self.widen_factor,
                dropRate=self.drop,
            )
        elif arch.startswith('resnet'):
            model = models.__dict__[arch](
                num_classes=self.num_classes,
                depth=self.depth,
            )
        else:
            model = models.__dict__[arch](num_classes=self.num_classes)

        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
        return model

    def _prepare_log(self, checkpoint, name, resume=False):
        title = 'cifar-10-' + self.arch + '-' + str(name)
        logger = Logger(os.path.join(checkpoint, 'log.txt'), title=title, resume=resume)
        if not resume:
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        return logger

    def train(self, trainloader, model, criterion, optimizer, epoch, use_cuda):
        # switch to train mode
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
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

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s ' \
                         '| Batch: {bt:.3f}s | Total: {total:} ' \
                         '| ETA: {eta:} | Loss: {loss:.4f} ' \
                         '| top1: {top1: .4f} | top5: {top5: .4f}' \
                         ''.format(batch=batch_idx + 1, size=len(trainloader), data=data_time.avg,
                                   bt=batch_time.avg, total=bar.elapsed_td,
                                   eta=bar.eta_td, loss=losses.avg,
                                   top1=top1.avg, top5=top5.avg)
            bar.next()
        bar.finish()
        return losses.avg, top1.avg

    def test(self, testloader, model, criterion, epoch, use_cuda):
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

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s ' \
                         '| Batch: {bt:.3f}s | Total: {total:} ' \
                         '| ETA: {eta:} | Loss: {loss:.4f} ' \
                         '| top1: {top1: .4f} | top5: {top5: .4f}' \
                         ''.format(batch=batch_idx + 1, size=len(testloader), data=data_time.avg,
                                   bt=batch_time.avg, total=bar.elapsed_td,
                                   eta=bar.eta_td, loss=losses.avg,
                                   top1=top1.avg, top5=top5.avg)
            bar.next()
        bar.finish()
        return losses.avg, top1.avg

    def save_checkpoint(self, states, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(checkpoint, filename)
        torch.save(states, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

    def load_checkpoint(self, checkpoint):
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'


    def adjust_learning_rate(self, optimizer, epoch):
        global state
        if epoch in self.schedule:
            state['lr'] *= self.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['lr']
