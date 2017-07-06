from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from models import resnet
from torch.optim import lr_scheduler

import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--lr_decay', type=str, default='80,120',
                        help='learning rate decay 0.1 when given epochs, e.g. 80,120.')
parser.add_argument('--momentum', type=float, default=0, metavar='M',
                    help='SGD momentum (default: 0)')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--gpus', type=str,
                       help='list of gpus to run, e.g. 0 or 0,2,5. empty means using gpu 0, -1 means using cpu.')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

if args.gpus is None or args.gpus is '':
    args.gpus = '0'
args.cuda = not args.gpus == '-1' and torch.cuda.is_available()
if args.cuda:
    devs = [int(i) for i in args.gpus.split(',')]
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

args.parallel = True if len(devs) > 1 else False
if args.parallel:
    import multiverso as mv
    from multiverso.torch_ext import torchmodel
    mv.init(sync=True, updater=b"sgd")   


kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                    transform=transforms.Compose([
			    transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor(),
			transforms.Normalize(mean = [x / 255 for x in [125.3, 123.0, 113.9]],
				std = [x / 255 for x in [63.0, 62.1, 66.7]])
                        ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
transforms.Normalize(mean = [x / 255 for x in [125.3, 123.0, 113.9]],
        std = [x / 255 for x in [63.0, 62.1, 66.7]])
                        ])),
    batch_size=args.batch_size, shuffle=False, **kwargs)


model = resnet.resnet20()
criterion = torch.nn.CrossEntropyLoss()

# if args.ngpu > 1:
#         model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
if args.parallel:
    model = torchmodel.MVTorchModel(model)

if args.cuda:
    device = devs[0] if args.parallel else devs[mv.worker_id()]
    model.cuda(device)
    criterion.cuda(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(i) for i in args.lr_decay.split(',')], gamma=0.1)

def train(epoch):
    model.train()
    scheduler.step()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(device), target.cuda(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def paralleltrain(epoch):
    model.train()
    scheduler.step()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx % mv.workers_num() != mv.worker_id():
            continue
        if args.cuda:
            data, target = data.cuda(device), target.cuda(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        model.cpu()
        model.mv_sync()
        model.cuda(device)

        if (batch_idx / mv.workers_num()) % args.log_interval == 0:
            print('Worker: {}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    mv.worker_id(), epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))

    if batch_idx % mv.workers_num() < mv.worker_id():
        optimizer.zero_grad()
        model.cpu()
        model.mv_sync()
        model.cuda(device)

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(device), target.cuda(device)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    if args.parallel:
        print('\nWorker: {}\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            mv.worker_id(), test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    else:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
 
for epoch in range(1, args.epochs + 1):
    paralleltrain(epoch) if args.parallel else train(epoch)
    test(epoch)
