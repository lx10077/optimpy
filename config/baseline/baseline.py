import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import argparse
import csv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models import *
from utils.common import prepare_dataset
from config.baseline.helper import make_train_path, mkdir
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from save_checkpoint')
parser.add_argument('--sess', default='cifar10_baseline', type=str, help='session id')
parser.add_argument('--seed', default=0, type=int, help='rng seed')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay (default=1e-4)')
args = parser.parse_args()
torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 128
base_learning_rate = 0.1

if use_cuda:
    # data parallel
    n_gpu = torch.cuda.device_count()
    batch_size *= n_gpu
    base_learning_rate *= n_gpu

# Data
trainloader, testloader, class_num = prepare_dataset(batch_size)
train_path = make_train_path()
checkpoint_folder = mkdir(os.path.join(train_path, 'checkpoint'))
exp_name = 'ckpt.t7.' + args.sess + '_' + str(args.seed)


# Model
if args.resume:
    # Load save_checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoint_folder), 'Error: no checkpoint directory found!'
    try:
        checkpoint = torch.load(os.path.join(checkpoint_folder, exp_name))
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    except Exception as e:
        raise Exception(e)
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    net = PreActResNet18(class_num)
    # net = ResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print('==> Using', torch.cuda.device_count(), 'GPUs.')
    cudnn.benchmark = True
    print('==> Using CUDA..')
else:
    print("==> Don't use CUDA..")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=base_learning_rate, momentum=0.9, weight_decay=args.decay)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = train_acc = train_total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # generate mixed inputs, two one-hot label vectors and mixing coefficient
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct = predicted.eq(targets).sum().item()
        target_size = targets.size(0)

        train_loss += loss.item()
        train_acc += correct
        train_total += target_size

        print('[Train] %d th [%d/%d] sLoss: %.3f | sAcc: %.4f%%' % (
            epoch, batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*train_acc/train_total))
        train_writer.add_scalar('train_loss', loss.item(), epoch * len(trainloader) + batch_idx)
        train_writer.add_scalar('train_acc', correct/target_size, epoch * len(trainloader) + batch_idx)
    return train_loss/len(trainloader), 100.*train_acc/train_total


# Testing
def test(epoch):
    global best_acc
    net.eval()
    test_loss = test_acc = test_total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs, 1)
            correct = predicted.eq(targets).sum().item()

            test_loss += loss.item()
            test_acc += correct
            test_total += targets.size(0)

            print('[Val]      [%d/%d] sLoss: %.3f | sAcc: %.4f%%' % (
                batch_idx, len(testloader), test_loss / (batch_idx + 1), 100. * test_acc / test_total))
            train_writer.add_scalar('val_loss', test_loss, epoch * len(testloader) + batch_idx)
            train_writer.add_scalar('val_acc', correct/targets.size(0), epoch * len(testloader) + batch_idx)

    # Save checkpoint.
    acc = 100.*test_acc/test_total
    if acc > best_acc:
        best_acc = acc
        save_checkpoint(acc, epoch)
    return test_loss/len(testloader), 100.*test_acc/test_total


def save_checkpoint(acc, epoch):
    # Save checkpoint.
    print('==> Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    save_path = os.path.join(checkpoint_folder, exp_name)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    torch.save(state, save_path)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = base_learning_rate
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (base_learning_rate - 0.1) * epoch / 10.
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


result_folder = os.path.join(train_path, 'results_')
train_event_folder = mkdir(os.path.join(train_path, 'train.event'))
val_event_folder = mkdir(os.path.join(train_path, 'val.event'))

logname = result_folder + net.__class__.__name__ + '_' + args.sess + '_' + str(args.seed) + '.csv'
train_writer = SummaryWriter(train_event_folder)
val_writer = SummaryWriter(val_event_folder)


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])


for epoch in range(start_epoch, 200):
    adjust_learning_rate(optimizer, epoch)
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
