from __future__ import division
import os, sys, shutil, time, random, math
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import models
import numpy as np
import random
import torch.nn.functional as F
from utils import _ECELoss

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Training script for Networks with Soft Sharing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data / Model
parser.add_argument('--data_path', metavar='DPATH', type=str, default='data', help='Path to dataset')
parser.add_argument('--dataset', metavar='DSET', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'], help='Choose between CIFAR/ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='wrn_r', help='model architecture: ' + ' | '.join(model_names) + ' (default: wide resnet)')
parser.add_argument('--width', type=int, metavar='N', default=10)
parser.add_argument('--num_nodes', type=int, metavar='N', default=8)
parser.add_argument('--affine', dest='affine', action='store_true', help='Enable learnable affine in bn')
parser.add_argument('--track_running_stats', dest='track_running_stats', action='store_true', help='Enable track_running_stats in bn')
parser.add_argument('--use_bn', action='store_true', default=False)

# Optimization
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')

#Regularization
parser.add_argument('--dropout_rate', type=float, default=0., help='dropout_rate.')
parser.add_argument('--droppath_rate', type=float, default=0.)
parser.add_argument('--learn_aggr', action='store_true')
parser.add_argument('--aux', action='store_true', default=False)

# Checkpoints
parser.add_argument('--save_path', type=str, default='/data/zhijie/snapshots_fbnn/', help='Folder to save checkpoints and log.')

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
parser.add_argument('--batch_arch', action='store_true', default=False)

# attack settings
parser.add_argument('--model_dirs', default=[], type=str, nargs='+')
parser.add_argument('--num_ensemble', type=int, default=1)
parser.add_argument('--attack_method', default='pgd', type=str)
parser.add_argument('--epsilon', default=8.0, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=5, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2.0, type=float,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')

args = parser.parse_args()
assert(len(args.model_dirs) > 0)
args.affine = True
args.track_running_stats = True
args.epsilon /= 255.
args.step_size /= 255.
if args.num_steps == 1:
    args.random = False
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()
job_id = args.job_id
args.save_path = args.save_path + job_id

out_str = str(args)
print(out_str)

if args.manualSeed is None: args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda: torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

num_stages = 3

def load_dataset():
    if args.dataset == 'cifar10':
        mean = torch.from_numpy(np.array([x / 255 for x in [125.3, 123.0, 113.9]])).view(1,3,1,1).cuda().float()
        std = torch.from_numpy(np.array([x / 255 for x in [63.0, 62.1, 66.7]])).view(1,3,1,1).cuda().float()
        dataset = dset.CIFAR10
        num_classes = 10
    elif args.dataset == 'cifar100':
        mean = torch.from_numpy(np.array([x / 255 for x in [129.3, 124.1, 112.4]])).view(1,3,1,1).cuda().float()
        std = torch.from_numpy(np.array([x / 255 for x in [68.2, 65.4, 70.4]])).view(1,3,1,1).cuda().float()
        dataset = dset.CIFAR100
        num_classes = 100
    elif args.dataset != 'imagenet': assert False, "Unknown dataset : {}".format(args.dataset)

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        test_transform = transforms.Compose([transforms.ToTensor()])
        test_data = dataset(args.data_path, train=False, transform=test_transform, download=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    else: assert False, 'Do not support dataset : {}'.format(args.dataset)

    return num_classes, test_loader, mean, std


def load_model(num_classes, log):
    net = models.__dict__[args.arch](args, num_classes)
    net = torch.nn.DataParallel(net.cuda(), device_ids=list(range(args.ngpu)))
    trainable_params = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([p.numel() for p in trainable_params])
    print_log("Number of parameters: {}".format(params), log)
    return net

def main():

    if not os.path.isdir(args.save_path): os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}_{}.txt'.format(args.manualSeed, args.attack_method)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("PyTorch  version : {}".format(torch.__version__), log)
    print_log("CuDNN  version : {}".format(torch.backends.cudnn.version()), log)

    if not os.path.isdir(args.data_path): os.makedirs(args.data_path)
    num_classes, test_loader, mean, std = load_dataset()

    models = []
    
    for model_dir in args.model_dirs:
        model_path = '/data/zhijie/snapshots_fbnn/' + model_dir + '/checkpoint.pth.tar'
        if not os.path.isfile(model_path):
            model_path = '/data/zhijie/snapshots_fbnn2/' + model_dir + '/checkpoint.pth.tar'
        print(model_path)
        checkpoint = torch.load(model_path)
        args.use_bn = False
        if '-bn' in model_path: args.use_bn = True
        net = load_model(num_classes, log)
        net.load_state_dict(checkpoint['state_dict'])
        # print(net.module.stage_1.adj.data[0].cpu().numpy())

        models.append(net.module)
    
    if args.attack_method == 'pgd':
        pgd(test_loader, mean, std, log, models)
    elif args.attack_method == 'ood':
        test_data1 = dset.SVHN(args.data_path, split='test', transform=transforms.Compose([transforms.ToTensor()]), download=True)
        test_loader1 = torch.utils.data.DataLoader(test_data1, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        test(test_loader1, mean, std, log, models, '_ood')
    elif args.attack_method == 'none':
        test(test_loader, mean, std, log, models)
    elif args.attack_method == 'div':
        test_snet_diversity(test_loader, mean, std, log, models)
    log.close()

def predict(X, mean, std, models):
    output = 0
    mi = 0
    for model in models:
        for j in range(args.num_ensemble):
            out_ = model(X.sub(mean).div(std)).softmax(-1)
            output += out_
            mi -= (- out_ * out_.log()).sum(1)
    output = output.div(args.num_ensemble*len(models))
    ent = (- output * output.log()).sum(1)
    mi = ent + mi.div(args.num_ensemble*len(models))
    return output, ent, mi

def grad(X, y, mean, std, models):
    probs = torch.zeros(args.num_ensemble*len(models), X.shape[0]).cuda()
    grads = torch.zeros([args.num_ensemble*len(models)] + list(X.shape)).cuda()

    for i, model in enumerate(models):
        for j in range(args.num_ensemble):
            with torch.enable_grad():
                X.requires_grad_()
                output = model(X.sub(mean).div(std))
                loss = F.cross_entropy(output, y, reduction='none')
                grad_ = torch.autograd.grad(
                    [loss], [X], grad_outputs=torch.ones_like(loss), retain_graph=False)[0].detach()
            grads[i*args.num_ensemble+j] = grad_
            probs[i*args.num_ensemble+j] = torch.gather(output.detach().softmax(-1), 1, y[:,None]).squeeze()
    probs /= probs.sum(0)
    grad_ = (grads * probs[:, :, None, None, None]).sum(0)
    return grad_

def _pgd_whitebox(X, y, mean, std, models,
                  epsilon, num_steps, step_size):

    out, ent, mi = predict(X, mean, std, models)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = X.clone()
    if args.random:
        X_pgd += torch.cuda.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon)

    for _ in range(num_steps):
        grad_ = grad(X_pgd, y, mean, std, models)

        X_pgd += step_size * grad_.sign()
        eta = torch.clamp(X_pgd - X, -epsilon, epsilon)
        X_pgd = torch.clamp(X + eta, 0, 1.0)
    out_pgd, ent_pgd, mi_pgd = predict(X_pgd, mean, std, models)
    err_pgd = (out_pgd.data.max(1)[1] != y.data).float().sum()
    print('err nat: ', err,'err pgd (white-box): ', err_pgd)
    return err, err_pgd, out, out_pgd, ent, ent_pgd, mi, mi_pgd

def pgd(val_loader, mean, std, log, models):
    print(args.epsilon, args.step_size, args.num_steps, args.random)
    top1_natural = AverageMeter()
    top5_natural = AverageMeter()
    top1_pgd = AverageMeter()
    top5_pgd = AverageMeter()
    entropies_natural, entropies_robust = [], []
    mis_natural, mis_robust = [], []

    for model in models:
        model.eval()
        if args.dropout_rate > 0.:
            for m in model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                  m.train()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            err_natural, err_robust, output_natual, output_robust, ent_natural, ent_robust, mi_natural, mi_robust = _pgd_whitebox(input, target, mean, std, models, args.epsilon, args.num_steps, args.step_size)

            prec1_natural, prec5_natural = accuracy(output_natual, target, topk=(1, 5))
            prec1_pgd, prec5_pgd = accuracy(output_robust, target, topk=(1, 5))

            top1_natural.update(prec1_natural.item(), input.size(0))
            top5_natural.update(prec5_natural.item(), input.size(0))

            top1_pgd.update(prec1_pgd.item(), input.size(0))
            top5_pgd.update(prec5_pgd.item(), input.size(0))

            entropies_natural.append(ent_natural)
            entropies_robust.append(ent_robust)
            mis_natural.append(mi_natural)
            mis_robust.append(mi_robust)

    np.save(args.save_path + "/entropies_natural.npy", torch.cat(entropies_natural, 0).data.cpu().numpy())
    np.save(args.save_path + "/entropies_robust_eps{}_ns{}_ss{}.npy".format(int(args.epsilon*255.), args.num_steps, int(args.step_size*255.)), torch.cat(entropies_robust, 0).data.cpu().numpy())
    np.save(args.save_path + "/mis_natural.npy", torch.cat(mis_natural, 0).data.cpu().numpy())
    np.save(args.save_path + "/mis_robust_eps{}_ns{}_ss{}.npy".format(int(args.epsilon*255.), args.num_steps, int(args.step_size*255.)), torch.cat(mis_robust, 0).data.cpu().numpy())
    print_log('  **Attack**  natural {top1_natural.avg:.3f} {top5_natural.avg:.3f} pgd {top1_pgd.avg:.3f} {top5_pgd.avg:.3f}'.format(top1_natural=top1_natural, top5_natural=top5_natural, top1_pgd=top1_pgd, top5_pgd=top5_pgd), log)

def test(val_loader, mean, std, log, models, suffix='_natural'):
    top1 = AverageMeter()
    top5 = AverageMeter()
    logits, labels, entropies, mis = [], [], [], []

    for model in models:
        model.eval()
        if args.dropout_rate > 0.:
            for m in model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                  m.train()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            out, ent, mi = predict(input, mean, std, models)
            prec1, prec5 = accuracy(out, target, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            logits.append(out.log())
            labels.append(target)
            entropies.append(ent)
            mis.append(mi)

    ece = _ECELoss()(torch.cat(logits, 0), torch.cat(labels, 0), os.path.join(args.save_path, 'cal{}.pdf'.format(suffix))).item()
    np.save(args.save_path + "/entropies{}.npy".format(suffix), torch.cat(entropies, 0).data.cpu().numpy())
    np.save(args.save_path + "/mis{}.npy".format(suffix), torch.cat(mis, 0).data.cpu().numpy())
    print_log(' {top1.avg:.4f} {top5.avg:.4f} {ece:.4f}'.format(top1=top1, top5=top5, ece=ece), log)

def test_snet_diversity(val_loader, mean, std, log, models):
    for model in models:
        model.eval()
    nums = 100
    with torch.no_grad():
        mses = [[] for _ in range(len(models))]
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)

            for j, model in enumerate(models):
                outputs = []
                for k in range(nums):
                    output1 = model(input.sub(mean).div(std)).softmax(-1)
                    outputs.append(output1)
                mse = []
                for k1 in range(nums):
                    for k2 in range(nums):
                        if k2 != k1:
                            mse.append(F.kl_div(outputs[k1].log(), outputs[k2], reduction='none').sum(1, keepdim=True))
                mse = torch.cat(mse, 1)
                mses[j].append(mse.data.cpu().numpy())
                print(i, j)

        x0 = val_loader.dataset.__getitem__(0)[0]
        x0 = x0.unsqueeze(0)
        outputs = []
        for k in range(nums):
            output1 = model(x0.sub(mean).div(std)).softmax(-1)
            outputs.append(output1)
        outputs = torch.cat(outputs)

    mses = [np.concatenate(mse) for mse in mses]
    for mse in mses:
        print(mse.max(1).mean(), mse.min(1).mean(), mse.mean())
    np.save(args.save_path + "/div_x{}.npy".format(0), outputs.data.cpu().numpy())

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

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

if __name__ == '__main__': main()
