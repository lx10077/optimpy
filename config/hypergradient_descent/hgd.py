import torch.nn as nn
import torch.backends.cudnn as cudnn

import os
import argparse
import csv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.common import prepare_dataset
from config.hypergradient_descent.helper import make_train_path, mkdir
from config.hypergradient_descent.model import *
from config.hypergradient_descent.optimizer import *
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Hypergradient descent PyTorch tests',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--device', help='selected CUDA device', default=0, type=int)
parser.add_argument('--seed', help='random seed', default=1, type=int)
parser.add_argument('--model', help='model (logreg, mlp, vgg)', default='logreg', type=str)
parser.add_argument('--method', help='method (sgd, sgd_hd, sgdn, sgdn_hd, adam, adam_hd)', default='adam', type=str)
parser.add_argument('--alpha_0', help='initial learning rate', default=0.001, type=float)
parser.add_argument('--beta', help='learning learning rate', default=0.000001, type=float)
parser.add_argument('--mu', help='momentum', default=0.9, type=float)
parser.add_argument('--weightDecay', help='regularization', default=1e-4, type=float)
parser.add_argument('--batchSize', help='minibatch size', default=128, type=int)
parser.add_argument('--epochs', help='stop after this many epochs (0: disregard)', default=2, type=int)
parser.add_argument('--iterations', help='stop after this many iterations (0: disregard)', default=0, type=int)
parser.add_argument('--lossThreshold', help='stop after reaching this loss (0: disregard)', default=0, type=float)
parser.add_argument('--silent', help='do not print output', action='store_true')
parser.add_argument('--parallel', help='parallelize', action='store_true')
parser.add_argument('--save', help='do not save output to file', action='store_true')
parser.add_argument('--workers', help='number of data loading workers', default=4, type=int)

parser.add_argument('--resume', '-r', action='store_true', help='resume from save_checkpoint')
parser.add_argument('--sess', default='hypergradient_descent', type=str, help='session id')
args = parser.parse_args()
args.task = 'mnist' if args.model == 'logreg' or args.model == 'mlp' else 'cifar10'


use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 128
torch.manual_seed(args.seed)


if use_cuda:
    # data parallel
    n_gpu = torch.cuda.device_count()
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True

# Data
trainloader, testloader, class_num = prepare_dataset(data_name=args.task,
                                                     batch_size=batch_size,
                                                     num_workers=args.workers)
train_path = make_train_path()
checkpoint_folder = mkdir(os.path.join(train_path, 'checkpoint'))
exp_name = 'ckpt.t7.{}_{}_{}_{}_{}_{}'.format(args.task, str(args.seed), args.model, args.method,
                                              args.alpha_0, args.beta)

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

    if args.model == 'logreg':
        net = LogReg(28 * 28, 10)
    elif args.model == 'mlp':
        net = MLP(28 * 28, 1000, 10)
    elif args.model == 'vgg':
        net = vgg16_bn()
        if args.parallel:
            net.features = torch.nn.DataParallel(net.features)
    else:
        raise Exception('Unknown model: {}'.format(args.model))


if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print('==> Using', torch.cuda.device_count(), 'GPUs.')
    cudnn.benchmark = True
    print('==> Using CUDA..')
else:
    print("==> Don't use CUDA..")

criterion = nn.CrossEntropyLoss()

if args.method == 'sgd':
    optimizer = SGD(net.parameters(), lr=args.alpha_0, weight_decay=args.weightDecay)
elif args.method == 'sgd_hd':
    optimizer = SGDHD(net.parameters(), lr=args.alpha_0, weight_decay=args.weightDecay,
                      hypergrad_lr=args.beta)
elif args.method == 'sgdn':
    optimizer = SGD(net.parameters(), lr=args.alpha_0, weight_decay=args.weightDecay,
                    momentum=args.mu, nesterov=True)
elif args.method == 'sgdn_hd':
    optimizer = SGDHD(net.parameters(), lr=args.alpha_0, weight_decay=args.weightDecay,
                      momentum=args.mu, nesterov=True,
                      hypergrad_lr=args.beta)
elif args.method == 'adam':
    optimizer = Adam(net.parameters(), lr=args.alpha_0, weight_decay=args.weightDecay)
elif args.method == 'adam_hd':
    optimizer = AdamHD(net.parameters(), lr=args.alpha_0, weight_decay=args.weightDecay,
                       hypergrad_lr=args.beta)
else:
    raise Exception('Unknown method: {}'.format(args.method))

print('==> Task: {}, Model: {}, Method: {}'.format(args.task, args.model, args.method))


# Training
def train(epoch, threshold_iter=0, threshold_loss=0):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = train_acc = train_total = 0
    train_alpha = iteration = 0
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
        alpha = optimizer.param_groups[0]['lr']
        target_size = targets.size(0)

        train_loss += loss.item()
        train_acc += correct
        train_alpha += alpha
        train_total += target_size
        iteration += 1

        print('[Train] %d th [%d/%d] sLoss: %.3f | sAcc: %.4f%% | sAlpha: %.5f%%' % (
            epoch, batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*train_acc/train_total,
            train_alpha/(batch_idx+1)))
        train_writer.add_scalar('train_loss', loss.item(), epoch * len(trainloader) + batch_idx)
        train_writer.add_scalar('train_acc', correct/target_size, epoch * len(trainloader) + batch_idx)
        train_writer.add_scalar('train_alpha', alpha/target_size, epoch * len(trainloader) + batch_idx)

        if threshold_iter != 0 and iteration > threshold_iter:
            print('==> Early stopping: iteration > {}'.format(args.iterations))
            break
        if threshold_loss >= 0 and loss <= threshold_loss:
            print('==> Early stopping: loss <= {}'.format(args.lossThreshold))
            break

    return train_loss/len(trainloader), 100.*train_acc/train_total, train_alpha/train_total


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


result_folder = os.path.join(train_path, 'results_')
train_event_folder = mkdir(os.path.join(train_path, 'train.event'))
val_event_folder = mkdir(os.path.join(train_path, 'val.event'))

logname = result_folder + exp_name + '.csv'
train_writer = SummaryWriter(train_event_folder)
val_writer = SummaryWriter(val_event_folder)


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'train acc', 'train alpha', 'test loss', 'test acc'])


for epoch in range(start_epoch, 200):
    train_loss, train_acc, train_alpha = train(epoch, args.iterations, args.lossThreshold)
    test_loss, test_acc = test(epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, train_acc, train_alpha, test_loss, test_acc])
