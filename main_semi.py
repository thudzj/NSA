''' 

python main_semi.py --num_ensemble 100 --dataset cifar10 --arch wrn_r --dropout_rate 0.  --learn_aggr --arch_seed_start 1 --arch_seed_end 5 --arch_p 0.7 --job-id snet-semi-1-ul20-decay5e-4 --epochs 100 --schedule 40 80 --decay 5e-4 --un_weight 20  --gammas 0.2 0.2 

python main_semi.py --dataset cifar10 --arch wrn_r --dropout_rate 0.2 --num_ensemble 100 --decay 5e-4 --learn_aggr --arch_type residual --job-id resnet-dp0.2-semi-1-ul20-decay5e-4 --epochs 100 --schedule 40 80 --un_weight 20 --gammas 0.2 0.2

main_semi.py --dataset cifar10 --arch wrn_r --dropout_rate 0 --num_ensemble 100 --learn_aggr --arch_type residual --job-id resnet-lbonly-1-decay5e-4 --epochs 100 --schedule 40 80 --decay 5e-4 --labeled_only --gammas 0.2 0.2
'''


from __future__ import division
import os, sys, shutil, time, random, math, copy
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, Cutout
import models
import numpy as np
import random
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from utils import _ECELoss

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Training script for Networks with Soft Sharing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data / Model
parser.add_argument('--data_path', metavar='DPATH', type=str, default='data', help='Path to dataset')
parser.add_argument('--dataset', metavar='DSET', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'], help='Choose between CIFAR/ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='wrn_r', help='model architecture: ' + ' | '.join(model_names) + ' (default: wide resnet)')
parser.add_argument('--width', type=int, metavar='N', default=10)
parser.add_argument('--num_nodes', type=int, metavar='N', default=8)
parser.add_argument('--use_bn', action='store_true', default=False)
parser.add_argument('--affine', dest='affine', action='store_true', help='Enable learnable affine in bn')
parser.add_argument('--track_running_stats', dest='track_running_stats', action='store_true', help='Enable track_running_stats in bn')

# Optimization
parser.add_argument('--epochs', metavar='N', type=int, default=150, help='Number of epochs to train.') #300
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--lr_scheduler', type=str, default='step', help='Lr_scheduler.')
parser.add_argument('--schedule', type=int, nargs='+', default=[45, 90, 120], help='Decrease learning rate at these epochs.') #[90, 180, 240]
parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2, 0.2], help='LR is multiplied by gamma on schedule')

#Regularization
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--cutout', dest='cutout', action='store_true', help='Enable cutout augmentation')
parser.add_argument('--dropout_rate', type=float, default=0., help='dropout_rate.')
parser.add_argument('--droppath_rate', type=float, default=0.)
parser.add_argument('--learn_aggr', action='store_true', default=False)
parser.add_argument('--aux', action='store_true', default=False)
parser.add_argument('--aux_weight', type=float, default=0.)
parser.add_argument('--label_smooth_rate', type=float, default=0.)

# Checkpoints
parser.add_argument('--save_path', type=str, default='/data/zhijie/snapshots_fbnn/', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='Path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='Evaluate model on test set')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers (default: 8)')

# Random seed
parser.add_argument('--manualSeed', type=int, default='1', help='manual seed')
parser.add_argument('--job-id', type=str, default='')

# arch_type
parser.add_argument('--arch_type', default='random', type=str)
parser.add_argument('--arch_seed_start', type=int, default=1)
parser.add_argument('--arch_seed_end', type=int, default=5)
parser.add_argument('--arch_p', type=float, default=0.7)
parser.add_argument('--gradient_est', default=None, type=str)
parser.add_argument('--arch_reg', type=float, default=1.)
parser.add_argument('--batch_arch', action='store_true', default=False)

# test settings
parser.add_argument('--num_ensemble', type=int, default=1)
parser.add_argument('--labeled_only', action='store_true', default=False)
parser.add_argument('--un_weight', type=float, default=20.)

args = parser.parse_args()
args.cutout = True
args.affine = True
args.track_running_stats = True
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()
job_id = args.job_id
args.save_path = args.save_path + job_id
result_png_path = os.path.join(args.save_path, 'curve.png')
result_cal_path = os.path.join(args.save_path, 'cal.pdf')
    
out_str = str(args)
print(out_str)

if args.manualSeed is None: args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda: torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

best_acc = 0
num_stages = 3

def load_dataset():
    if args.dataset == 'cifar10':
        mean, std = [x / 255 for x in [125.3, 123.0, 113.9]],  [x / 255 for x in [63.0, 62.1, 66.7]]
        dataset_twice = CIFAR10_twice
        dataset = dset.CIFAR10
        num_classes = 10
    elif args.dataset == 'cifar100':
        mean, std = [x / 255 for x in [129.3, 124.1, 112.4]], [x / 255 for x in [68.2, 65.4, 70.4]]
        dataset_twice = CIFAR100_twice
        dataset = dset.CIFAR100
        num_classes = 100

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)])
        if args.cutout: train_transform.transforms.append(Cutout(n_holes=1, length=16))
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        train_data = dataset_twice(args.data_path, train=True, transform=train_transform, download=True)

        indices = list(range(len(train_data)))
        np.random.shuffle(indices)
        split = int(4000 if args.dataset == 'cifar10' else 10000)
        la_indices, un_indices = indices[:split], indices[split:]

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, sampler=SubsetRandomSampler(la_indices), drop_last=True)
        train_loader_un = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, sampler=SubsetRandomSampler(un_indices), drop_last=True)
        print(len(train_loader), len(train_loader_un))

        test_data = dataset(args.data_path, train=False, transform=test_transform, download=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    else: assert False, 'Do not support dataset : {}'.format(args.dataset)

    return num_classes, DataIter(train_loader), DataIter(train_loader_un), test_loader


def load_model(num_classes, log):
    net = models.__dict__[args.arch](args, num_classes)
    net = torch.nn.DataParallel(net.cuda(), device_ids=list(range(args.ngpu)))
    trainable_params = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([p.numel() for p in trainable_params])
    print_log("Number of parameters: {}".format(params), log)
    return net

def main():
    global best_acc

    if not os.path.isdir(args.save_path): os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}{}.txt'.format(args.manualSeed, '_eva' if args.evaluate else '')), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("PyTorch  version : {}".format(torch.__version__), log)
    print_log("CuDNN  version : {}".format(torch.backends.cudnn.version()), log)

    if not os.path.isdir(args.data_path): os.makedirs(args.data_path)

    num_classes, train_iter, train_un_iter, test_loader = load_dataset()

    net = load_model(num_classes, log)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    if args.label_smooth_rate > 0.:
        criterion_train = LabelSmoothingLoss(num_classes, args.label_smooth_rate)
    else:
        criterion_train = criterion
    params = group_weight_decay(net, state['decay'])
    optimizer = torch.optim.SGD(params, state['learning_rate'], momentum=state['momentum'], nesterov=(state['momentum'] > 0.0))

    recorder = RecorderMeter(args.epochs)
    if args.resume:
        if args.resume == 'auto': args.resume = os.path.join(args.save_path, 'checkpoint.pth.tar')
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            recorder.refresh(args.epochs)
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = recorder.max_accuracy(False)
            print_log("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(args.resume, best_acc, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint", log)

    if args.evaluate:
        validate(test_loader, net, criterion, log, args.num_ensemble)
        return

    start_time = time.time()
    epoch_time = AverageMeter()
    train_los = -1

    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule, train_los)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                    + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        train_acc, train_los = train(train_iter, train_un_iter, net, criterion_train, optimizer, epoch, log)

        val_acc, val_los   = validate(test_loader, net, criterion, log, 1 if epoch < args.epochs -1 else args.num_ensemble)
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)

        is_best = False
        if val_acc > best_acc:
            is_best = True
            best_acc = val_acc    

        save_dict = {
          'epoch': epoch + 1,
          'state_dict': net.state_dict(),
          'recorder': recorder,
          'optimizer' : optimizer.state_dict(),
        }
        save_checkpoint(save_dict, False, args.save_path, 'checkpoint.pth.tar')
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(result_png_path)
    log.close()


def train(train_iter, train_un_iter, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_un = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    arch_prob = []

    model.train()

    end = time.time()
    len_iter = len(train_un_iter.data_loader)
    criterion_mse = nn.MSELoss(size_average=False).cuda()
    for i in range(len_iter):
        input, target, input1 = train_iter.next_batch
        input_un, _, input_un1 = train_un_iter.next_batch
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)

        output_all = model(torch.cat([input, input_un]))
        with torch.no_grad():
            output_all1 = model(torch.cat([input1, input_un1]))
        output = output_all[:args.batch_size]

        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))
        loss_un = criterion_mse(output_all, output_all1)/float(output_all.shape[1]*args.batch_size*2)
        losses_un.update(loss_un.item(), input.size(0))
        
        weight_cl = 0 if args.labeled_only else cal_consistency_weight(epoch*len_iter+i, end_ep=(args.epochs//2)*len_iter, end_w=1.0)
        loss += weight_cl * args.un_weight * loss_un

        optimizer.zero_grad()
        loss.backward()
    
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i == len_iter-1:
            print_log('  Epoch: [{:03d}]   '
                        'Time {batch_time.avg:.3f}   '
                        'Data {data_time.avg:.3f}   '
                        'Loss {loss.avg:.4f}   '
                        'Loss un {loss_un.avg:.4f}   '
                        'Prec@1 {top1.avg:.3f}   '
                        'Prec@5 {top5.avg:.3f}   '.format(
                        epoch, batch_time=batch_time, data_time=data_time, loss_un=losses_un,
                        loss=losses, top1=top1, top5=top5) + time_string(), log)
            if len(arch_prob) > 0: print_log(np.array_repr(np.stack(arch_prob).sum(0)/50000.), log)
    return top1.avg, losses.avg

def validate(val_loader, model, criterion, log, num_ensemble):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    if args.dropout_rate > 0. and num_ensemble > 1:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
              m.train()

    entropies = []
    logits = []
    labels = []

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            output = 0
            for j in range(num_ensemble):
                output += model(input).softmax(-1)
            output = output.div(num_ensemble).log()
            # output = model(input)
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            logits.append(output)
            labels.append(target)
            entropies.append((- output * output.exp()).sum(1))

        entropies = torch.cat(entropies, 0).data.cpu().numpy()
        ece = _ECELoss()(torch.cat(logits, 0), torch.cat(labels, 0), result_cal_path).item()

    np.save(args.save_path + "/logits.npy", torch.cat(logits, 0).data.cpu().numpy())
    np.save(args.save_path + "/entropies.npy", entropies)

    print_log('  **Test**  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss {losses.avg:.5f} ECE {ece:.5f} '.format(top1=top1, top5=top5, error1=100-top1.avg, losses=losses, ece=ece), log)
    return top1.avg, losses.avg

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule, loss):
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step): lr = lr * gamma
        else: break
    for param_group in optimizer.param_groups: param_group['lr'] = lr
    return lr

def group_weight_decay(net, weight_decay, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue
        if sum([pattern in name for pattern in skip_list]) > 0: no_decay.append(param)
        else: decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]

def accuracy(output, target, topk=(1,)):
    if len(target.shape) > 1: return torch.tensor(1), torch.tensor(1)
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

class CIFAR10_twice(dset.CIFAR10):
    def __init__(self, root, train, transform, download):
        super(CIFAR10_twice, self).__init__(root, train=train, transform=transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img1 = np.copy(img)
        img = Image.fromarray(img)
        img1 = Image.fromarray(img1)
        if self.transform is not None:
            img = self.transform(img)
            img1 = self.transform(img1)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, img1

class CIFAR100_twice(dset.CIFAR100):
    def __init__(self, root, train, transform, download):
        super(CIFAR100_twice, self).__init__(root, train=train, transform=transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img1 = np.copy(img)
        img = Image.fromarray(img)
        img1 = Image.fromarray(img1)
        if self.transform is not None:
            img = self.transform(img)
            img1 = self.transform(img1)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, img1

def cal_consistency_weight(epoch, init_ep=0, end_ep=150, init_w=0.0, end_w=20.0):
    """Sets the weights for the consistency loss"""
    if epoch > end_ep:
        weight_cl = end_w
    elif epoch < init_ep:
        weight_cl = init_w
    else:
        T = float(epoch - init_ep)/float(end_ep - init_ep)
        #weight_mse = T * (end_w - init_w) + init_w #linear
        weight_cl = (math.exp(-5.0 * (1.0 - T) * (1.0 - T))) * (end_w - init_w) + init_w #exp
    #print('Consistency weight: %f'%weight_cl)
    return weight_cl

class DataIter(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._data_iter = None

    @property
    def next_batch(self):
        if self._data_iter is None:
            self._data_iter = iter(self.data_loader)
        try:
            data, label, data1 = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.data_loader)
            data, label, data1 = next(self._data_iter)
        return data, label, data1

if __name__ == '__main__': main()
