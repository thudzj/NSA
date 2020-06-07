'''

python attack.py --job-id test --model_dirs residual-dp0.2-1-decay3 --droppath_rate 0.2 --attack_method none --num_ensemble 100
96.9400 99.9200 0.0148

python attack.py --job-id residual-do0.2-la-decay3 --model_dirs residual-do0.2-1-la-decay3 residual-do0.2-2-la-decay3 residual-do0.2-3-la-decay3 residual-do0.2-4-la-decay3 residual-do0.2-5-la-decay3 residual-do0.2-6-la-decay3 residual-do0.2-7-la-decay3 --dropout_rate 0.2 --attack_method none --num_ensemble 10 --learn_aggr
97.2300 99.9600 0.0028

python attack.py --job-id random-p0.6-dp0.2-decay3 --model_dirs random-p0.6-dp0.2-13-decay3 random-p0.6-dp0.2-12-decay3 random-p0.6-dp0.2-14-decay3 random-p0.6-dp0.2-15-decay3 random-p0.6-dp0.2-16-decay3 random-p0.6-dp0.2-17-decay3 random-p0.6-dp0.2-18-decay3 --droppath_rate 0.2 --num_ensemble 10 --learn_aggr  --attack_method none
97.2900 99.9700 0.0045

ensemble MC dropout
python attack.py --job-id residual-do0.2-la-decay5 --model_dirs residual-do0.2-1-la-decay5 residual-do0.2-2-la-decay5 residual-do0.2-3-la-decay5 residual-do0.2-4-la-decay5 residual-do0.2-5-la-decay5 residual-do0.2-6-la-decay5 residual-do0.2-7-la-decay5 --dropout_rate 0.2 --attack_method none --num_ensemble 10 --learn_aggr
97.2200 99.9700 0.0040
ood:  11.9852 52.5008 0.4848      0.9805840945759065 (auc)
pgd4-5-1: 9.440 63.700            0.51748735 
pgd3-4-1: 20.140 80.210           0.66599435
pgd2-3-1: 39.340 92.410           0.78871983
pgd1-2-1: 70.940 99.030           0.809010485

ensemble random-topo
python attack.py --job-id random-p0.6-dp0.2-decay5 --model_dirs random-p0.6-dp0.2-21-decay5 random-p0.6-dp0.2-22-decay5 random-p0.6-dp0.2-23-decay5 random-p0.6-dp0.2-24-decay5 random-p0.6-dp0.2-25-decay5 random-p0.6-dp0.2-26-decay5 random-p0.6-dp0.2-27-decay5 --droppath_rate 0.2 --num_ensemble 10 --learn_aggr  --attack_method none
97.4500 99.9600 0.0032
ood: 12.0928 51.5865 0.4487       0.9763901621081745
pgd4-5-1: 13.210 64.990           0.60546179
pgd3-4-1: 24.220 81.090           0.720793155
pgd2-3-1: 42.520 92.940           0.8151670550000001
pgd1-2-1: 71.690 98.870           0.8287523700000001

without dp:
python attack.py --job-id random-p0.6 --model_dirs random-p0.6-1 random-p0.6-2 random-p0.6-3 random-p0.6-21 random-p0.6-22 random-p0.6-23 --droppath_rate 0. --num_ensemble 1 --learn_aggr  --attack_method none
97.5200 99.9600 0.0040

deep ensemble
python attack.py --job-id residual-deepensemble-la-decay5 --model_dirs residual-do0.2-1-la-decay5 residual-do0.2-2-la-decay5 residual-do0.2-3-la-decay5 residual-do0.2-4-la-decay5 residual-do0.2-5-la-decay5 residual-do0.2-6-la-decay5 residual-do0.2-7-la-decay5 --dropout_rate 0. --attack_method none --num_ensemble 1 --learn_aggr
97.2500 99.9500 0.0078
ood: 11.8316 52.2895 0.4737       0.9850332821143208
pgd4-5-1: 14.070 69.020           0.51455499
pgd3-4-1: 26.930 83.320           0.66230119
pgd2-3-1: 46.720 93.280           0.79021222
pgd1-2-1: 74.460 98.920           0.8229056299999999

MC dropout
python attack.py --job-id residual-do0.2-4-la-decay5 --model_dirs  residual-do0.2-4-la-decay5 --dropout_rate 0.2 --attack_method none --num_ensemble 70 --learn_aggr
96.7900 99.9500 0.0103
ood: 10.9365 50.5148 0.6519       0.924133205285802
pgd4-5-1: 4.850 56.200            0.340089315
pgd3-4-1: 13.290 73.250           0.495070005
pgd2-3-1: 30.870 88.750           0.6798693050000001
pgd1-2-1: 62.420 97.950           0.78640317


cifar100:
ensemble MC dropout
CUDA_VISIBLE_DEVICES=6 python attack.py --job-id residual-do0.2-la-cifar100-decay5 --model_dirs residual-do0.2-1-la-cifar100-decay5 residual-do0.2-2-la-cifar100-decay5 residual-do0.2-3-la-cifar100-decay5 residual-do0.2-4-la-cifar100-decay5 residual-do0.2-5-la-cifar100-decay5 residual-do0.2-6-la-cifar100-decay5 residual-do0.2-7-la-cifar100-decay5 --dropout_rate 0.2 --attack_method none --num_ensemble 10 --learn_aggr --dataset cifar100
83.9600 96.6000 0.0182
ood: 0.0691 1.2792 0.4613     0.915634617009834 (auc)
pgd4-5-1: 4.790 31.230        0.484250675
pgd3-4-1: 9.900 41.570        0.582381855
pgd2-3-1: 21.570 58.540       0.6730973200000001
pgd1-2-1: 46.340 80.870       0.69465719

ensemble random-topo
python attack.py --job-id random-p0.6-dp0.2-cifar100-decay5 --model_dirs random-p0.6-dp0.2-21-cifar100-decay5 random-p0.6-dp0.2-22-cifar100-decay5 random-p0.6-dp0.2-23-cifar100-decay5 random-p0.6-dp0.2-24-cifar100-decay5 random-p0.6-dp0.2-25-cifar100-decay5 random-p0.6-dp0.2-26-cifar100-decay5 random-p0.6-dp0.2-27-cifar100-decay5 --droppath_rate 0.2 --num_ensemble 10 --learn_aggr  --attack_method none --dataset cifar100
84.2600 96.8600 0.0199
ood: 0.1460 2.0897 0.4490     0.9100480466349109
pgd4-5-1: 5.800 28.940        0.56160467
pgd3-4-1: 11.260 39.580       0.643273105
pgd2-3-1: 22.290 56.090       0.7158432849999999
pgd1-2-1: 45.410 78.920       0.73727352


mc Dropout
python attack.py --job-id residual-do0.2-4-la-cifar100-decay5 --model_dirs residual-do0.2-4-la-cifar100-decay5 --dropout_rate 0.2 --attack_method none --num_ensemble 70 --learn_aggr --dataset cifar100
81.8100 95.7100 0.0360
ood: 0.1460 1.7286 0.5735     0.7678129494468348
pgd4-5-1: 2.430 27.840        0.17530188000000002
pgd3-4-1: 5.580 36.650        0.28629878
pgd2-3-1: 13.220 52.040       0.449471455
pgd1-2-1: 34.790 75.080       0.6039910199999999

deep ensemble:
python attack.py --job-id residual-deepensemble-la-cifar100-decay5 --model_dirs residual-do0.2-1-la-cifar100-decay5 residual-do0.2-2-la-cifar100-decay5 residual-do0.2-3-la-cifar100-decay5 residual-do0.2-4-la-cifar100-decay5 residual-do0.2-5-la-cifar100-decay5 residual-do0.2-6-la-cifar100-decay5 residual-do0.2-7-la-cifar100-decay5 --dropout_rate 0 --attack_method none --num_ensemble 1 --learn_aggr --dataset cifar100
84.1400 96.6200 0.0181
ood: 0.0960 1.6864 0.5224     0.9080699024277812
pgd4-5-1: 6.060 35.000        0.49811062
pgd3-4-1: 11.920 45.870       0.5882177049999999
pgd2-3-1: 24.290 61.230       0.68356626
pgd1-2-1: 47.680 81.190       0.7225549600000001


————————————————————————————————————————————————

python attack.py --job-id random-p0.8 --model_dirs random-p0.8-1 random-p0.8-2 random-p0.8-3 random-p0.8-4 random-p0.8-5 random-p0.8-6 random-p0.8-7  --droppath_rate 0. --num_ensemble 1 --learn_aggr  --attack_method none (gpu36)
97.5600 99.9700 0.0023

python attack.py --job-id random-p0.7 --model_dirs random-p0.7-1 random-p0.7-2 random-p0.7-3 random-p0.7-4 random-p0.7-5 random-p0.7-6 random-p0.7-7  --droppath_rate 0. --num_ensemble 1 --learn_aggr  --attack_method none (gpu36)
97.4600 99.9600 0.0036

python attack.py --job-id random-p0.6 --model_dirs random-p0.6-1 random-p0.6-2 random-p0.6-3 random-p0.6-4 random-p0.6-5 random-p0.6-6 random-p0.6-7  --num_ensemble 1 --learn_aggr  --attack_method none (gpu29)
97.4300 99.9400 0.0028

python attack.py --job-id residual --model_dirs residual-1 residual-2 residual-3 residual-4 residual-5 residual-6 residual-7  --num_ensemble 1 --learn_aggr  --attack_method none (gpu32)
97.4100 99.9600 0.0042


----cifar100
python attack.py --job-id random-cifar100-p0.8 --model_dirs random-cifar100-p0.8-1 random-cifar100-p0.8-2 random-cifar100-p0.8-3 random-cifar100-p0.8-4 random-cifar100-p0.8-5 random-cifar100-p0.8-6 random-cifar100-p0.8-7 --num_ensemble 1 --learn_aggr  --attack_method none --dataset cifar100 (gpu29)
85.0000 96.9600 0.0412

python attack.py --job-id residual-cifar100 --model_dirs residual-cifar100-1 residual-cifar100-2 residual-cifar100-3 residual-cifar100-4 residual-cifar100-5 residual-cifar100-6 residual-cifar100-7 --num_ensemble 1 --learn_aggr  --attack_method none --dataset cifar100 (gpu29)
84.7400 96.8000 0.0214





______________________________________________________________________________new logs:

python attack.py --job-id resnet-r-aux0.1-long-one --model_dirs resnet-r-aux0.1-long  --droppath_rate 0. --num_ensemble 1 --aux --learn_aggr --arch_type residual --attack_method ood
    ood: 12.5653 55.4932 0.7451         0.5
    pgd1-2-1: 63.870 97.970             0.5
    pgd2-3-1: 34.800 91.840             0.5
    pgd3-4-1: 18.680 83.150             0.5
    pgd4-5-1: 9.380 73.300              0.5
    fgsm8: 49.320 91.140                0.5
    pgd8-20-1: 0.020 47.980             0.5

python attack.py --job-id resnet-r-aux0.1-dp0.2-long --model_dirs resnet-r-aux0.1-dp0.2-long --dropout_rate 0.2 --num_ensemble 100 --aux --learn_aggr --arch_type residual --attack_method ood
    ood: 12.3963 51.0141 0.6588         0.9345764366933005
    pgd1-2-1: 64.240 98.550             0.75450405
    pgd2-3-1: 34.540 92.260             0.69417146
    pgd3-4-1: 18.270 82.890             0.5637272
    pgd4-5-1: 9.360 73.080              0.44944654500000003
    fgsm8: 38.530 85.300                0.8794589299999999

python attack.py --job-id resnet-r-aux0.1-long --model_dirs resnet-r-aux0.1-long resnet-r-aux0.1-long_2 resnet-r-aux0.1-long_3 resnet-r-aux0.1-long_4 resnet-r-aux0.1-long_5 --droppath_rate 0. --num_ensemble 1 --aux --learn_aggr --arch_type residual --attack_method  ood
    ood: 13.7139 51.7325 0.4938         0.977136151659496
    pgd1-2-1: 74.780 99.020             0.810345195
    pgd2-3-1: 45.760 95.060             0.8297494600000002
    pgd3-4-1: 25.930 87.820             0.769018275
    pgd4-5-1: 14.290 78.990             0.687674485
    fgsm8: 45.110 89.350                0.92005845
    pgd8-20-1: 0.020 43.500             0.36622261

python attack.py --job-id snet-r-p0.7-aux0.1-long-ens5 --model_dirs snet-1_1-r-p0.7-aux0.1-long snet-2_2-r-p0.7-aux0.1-long snet-3_3-r-p0.7-aux0.1-long snet-4_4-r-p0.7-aux0.1-long snet-5_5-r-p0.7-aux0.1-long --droppath_rate 0. --num_ensemble 1 --aux --learn_aggr --arch_type random --arch_seed_end 1 --attack_method ood
    ood: 14.6973 51.4597 0.5484         0.9625310483251382
    pgd1-2-1: 73.560 99.070             0.8403298450000001
    pgd2-3-1: 42.930 95.420             0.8500800049999999
    pgd3-4-1: 21.940 87.780             0.7629206400000001
    pgd4-5-1: 9.750 77.670              0.65931777
    fgsm8: 35.160 88.500                0.93870785
    pgd8-20-1: 0.000 37.430             0.278706055

python attack.py --job-id snet-1_5-r-p0.7-aux0.1-long --model_dirs  snet-1_5-r-p0.7-aux0.1-long --droppath_rate 0. --num_ensemble 100 --aux --learn_aggr  --attack_method ood
    ood: 14.0481 51.8016 0.5211         0.9702114781807005
    pgd1-2-1: 62.950 97.920             0.73693282
    pgd2-3-1: 40.050 92.970             0.704751995
    pgd3-4-1: 26.290 86.490             0.617586785
    pgd4-5-1: 17.020 80.570             0.521889285
    fgsm8: 52.890 90.330                0.836784755

CUDA_VISIBLE_DEVICES=2 python attack.py --arch wrn_r_d --job-id snet-1_5-r-p0.7-aux0.1-long-st --model_dirs  snet-1_5-r-p0.7-aux0.1-long-st --droppath_rate 0. --num_ensemble 100 --aux --learn_aggr --gradient_est st --attack_method ood (use seed 2 may be better)
    ood: 13.6140 52.9502 0.5010         0.9675918753841426
    pgd1-2-1: 64.110 98.360             0.7268547649999999
    pgd2-3-1: 39.500 93.520             0.6568236
    pgd3-4-1: 25.630 87.360             0.531518535
    pgd4-5-1: 16.440 80.990             0.40554362
    fgsm8: 54.270 90.990                0.8177267699999999

-----------------
div 
python attack.py --job-id test --model_dirs  snet-1_50-r-p0.7-aux0.1-long snet-1_50-r-p0.7-aux0.1-long-bn --droppath_rate 0. --num_ensemble 100 --aux --learn_aggr  --attack_method div --arch_seed_end 50
  100: 0.06791054844101772, 0.03486793362122925
  50: 0.04925205537318429, 0.031036703611570824
  20: 0.03388627751765749, 0.024979660381810573
  10: 0.024629329967819438, 0.02187615048281754 
  5: 0.02163786566639437, 0.016208570049721982         max: 0.06594446663379291, 0.04872640388534416


snet-1_5000-r-p0.7-aux0.1-long-bn-noaggr-ba    18.78055 1.8058314e-05 0.544069
snet-1_500-r-p0.7-aux0.1-long-bn-noaggr-ba     17.861925 -1.5191988e-08 0.48466486
snet-1_50-r-p0.7-aux0.1-long-bn-noaggr-ba      9.796614 -1.399474e-09 0.5035083
snet-1_5-r-p0.7-aux0.1-long-bn-noaggr-ba       1.9556723 0.0 0.37045115


snet-1_5000-r-p0.7-aux0.1-long-bn-noaggr       1.383051 1.1879978e-05 0.05353544
snet-1_500-r-p0.7-aux0.1-long-bn-noaggr        1.356035 -3.14137e-08 0.053081423
snet-1_100-r-p0.7-aux0.1-long-bn-noaggr        0.74436593 -2.6291183e-09 0.044343702
snet-1_50-r-p0.7-aux0.1-long-bn-noaggr         0.47241563 -2.521574e-13 0.038251754
snet-1_5-r-p0.7-aux0.1-long-bn-noaggr          0.10132014 0.0 0.0222339

snet-1_5000-r-p0.7-aux0.1-long-bn              0.73616403 1.4610632e-05 0.043269455
snet-1_500-r-p0.7-aux0.1-long-bn               0.63129395 -1.3852028e-10 0.03980561
snet-1_50-r-p0.7-aux0.1-long-bn                0.3565322 0.0 0.030826781
snet-1_5-r-p0.7-aux0.1-long-bn                 0.07142312 0.0 0.015971957

snet-1_500-r-p0.7-aux0.1-long                  2.812638 0.0 0.14809568
snet-1_100-r-p0.7-aux0.1-long                  0.8518037 -1.42624e-11 0.06873329
snet-1_50-r-p0.7-aux0.1-long                   0.5393519 0.0 0.048504125
snet-1_5-r-p0.7-aux0.1-long                    0.09492776 0.0 0.021292947


'''
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
