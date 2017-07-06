
import argparse

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
args.cuda = not args.gpus == '-1'
if args.cuda:
    devs = [int(i) for i in args.gpus.split(',')]


args.parallel = True if len(devs) > 1 else False

if args.parallel:
    import multiverso as mv
    from multiverso.torch_ext import torchmodel
    mv.init(sync=True, updater=b"sgd")  
    mv.shutdown()


