import math
import torch
import torch.nn as nn
from torch.nn import init
from functools import partial
import numpy as np

class ArchConditionalBatchNorm(torch.nn.Module):
    def __init__(self, num_features, arch_nums, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(ArchConditionalBatchNorm, self).__init__()
        self.num_features = num_features
        self.arch_nums = arch_nums
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(arch_nums, num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(arch_nums, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

    def forward(self, input, arch_ids):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        out = torch.nn.functional.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        if self.affine:
            shape = [input.size(0), self.num_features] + (input.dim() - 2) * [1]
            out = torch.addcmul(self.bias[arch_ids].view(shape), self.weight[arch_ids].view(shape), out)
        return out

    def extra_repr(self):
        return '{num_features}, arch_nums={arch_nums}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

class NormFunc(nn.Module):
    def __init__(self, args, num_features, arch_nums=1):
        super(NormFunc, self).__init__()
        if arch_nums == 1 or args.use_bn:
            self.fn = nn.BatchNorm2d(num_features,affine=args.affine,track_running_stats=args.track_running_stats)
            self.use_bn = True
        else:
            self.fn = ArchConditionalBatchNorm(num_features, arch_nums, 
                    affine=args.affine,track_running_stats=args.track_running_stats)
            self.use_bn = False

    def forward(self, x, arch_ids=None):
        if self.use_bn:
            return self.fn(x)
        else:
            return self.fn(x, arch_ids)

class ReLuConvBN(nn.Module):
  def __init__(self, args, inplanes, outplanes, kernel_size=3, stride=1, padding=1, arch_nums=1):
    super(ReLuConvBN, self).__init__()
    self.relu = nn.ReLU(inplace=True)
    self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    self.dropout = nn.Dropout(p=args.dropout_rate)
    self.norm = NormFunc(args, outplanes, arch_nums)

  def forward(self, x, arch_ids=None):
    return self.norm(self.dropout(self.conv(self.relu(x))), arch_ids)

class Transition(nn.Module):
  def __init__(self, args, inplanes, outplanes, arch_nums=1):
    super(Transition, self).__init__()
    self.relu = nn.ReLU(inplace=True)
    self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    self.norm = NormFunc(args, outplanes, arch_nums)

  def forward(self, x, arch_ids=None):
    return self.norm(self.pool(self.conv(self.relu(x))), arch_ids)

class Stage(nn.Module):
  def __init__(self, args, inplanes, adjacencies):
    super(Stage, self).__init__()
    self.num_nodes = args.num_nodes
    self.learn_aggr = args.learn_aggr
    adj = torch.from_numpy(np.stack(adjacencies))
    adj = adj.float() / adj.sum(2, keepdim=True)
    self.adj = nn.Parameter(adj, requires_grad=False)
    if self.learn_aggr:
        self.mean_weight = nn.Parameter(torch.ones(self.adj.shape[0], self.num_nodes+1, self.num_nodes+1))
        self.sigmoid = nn.Sigmoid()
    
    self.nodeop = nn.ModuleList()
    for _ in range(self.num_nodes):
        self.nodeop.append(ReLuConvBN(args, inplanes, inplanes, arch_nums=self.adj.shape[0]))

  def forward(self, x, arch_ids):
    if self.learn_aggr:
        mask = (self.adj * self.sigmoid(self.mean_weight))[arch_ids, :, :]
    else:
        mask = self.adj[arch_ids, :, :]

    outputs = [x]
    for i in range(self.num_nodes):
        input = sum(outputs[j]*mask[:, i:(i+1), j:(j+1), None].contiguous() for j in range(i+1))
        outputs.append(self.nodeop[i](input, arch_ids))
    output = sum(outputs[j]*mask[:, -1:, j:(j+1), None].contiguous() for j in range(self.num_nodes+1))
    return output

class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x

class WRN(nn.Module):
    def __init__(self, args, num_classes=10):
        super(WRN, self).__init__()
        n_channels = [16*args.width, 32*args.width, 64*args.width]
        self.num_classes = num_classes
        self.batch_arch = args.batch_arch

        self.arch_nums = args.arch_seed_end+1-args.arch_seed_start
        adjacencies = [[], [], []]
        if args.arch_type == 'random':
            for i in range(args.arch_seed_start, args.arch_seed_end+1):
                rng = np.random.RandomState(i)
                for j in range(3):
                    adj_one = np.tril(rng.rand(args.num_nodes+1, args.num_nodes+1) > args.arch_p, -1).astype(np.int) + np.eye(args.num_nodes+1).astype(np.int)
                    adjacencies[j].append(adj_one)
            print(adjacencies[0][0])
        elif args.arch_type == 'residual':
            self.arch_nums = 1
            adj_one = torch.eye(args.num_nodes+1)
            for i in range(args.num_nodes+1):
                if i % 2 == 0:
                    for j in range(i, args.num_nodes+1, 2):
                        adj_one[j, i] = 1
            print(adj_one)
            adjacencies[0].append(adj_one)
            adjacencies[1].append(adj_one)
            adjacencies[2].append(adj_one)

        self.conv_3x3 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = NormFunc(args, n_channels[0], self.arch_nums)
        self.stage_1 = Stage(args, n_channels[0], adjacencies[0])
        self.tran_1 = Transition(args, n_channels[0], n_channels[1], self.arch_nums)
        self.stage_2 = Stage(args, n_channels[1], adjacencies[1])
        self.tran_2 = Transition(args, n_channels[1], n_channels[2], self.arch_nums)
        self.stage_3 = Stage(args, n_channels[2], adjacencies[2])
        self.classifier = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(8), nn.Flatten(), nn.Linear(n_channels[2], num_classes))

        self.aux = args.aux
        if self.aux:
            self.auxiliary_head = AuxiliaryHeadCIFAR(n_channels[2], num_classes)

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if not 'se' in name:
                    init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x, arch=None):
        if arch:
            arch_ids = torch.cuda.LongTensor(x.shape[0]).fill_(arch)
        else:
            if self.batch_arch:
                arch_ids = torch.cuda.LongTensor(x.shape[0]).fill_(np.random.randint(0, self.arch_nums))
            else:
                arch_ids = torch.cuda.LongTensor(x.shape[0]).random_(0, self.arch_nums)
        
        x = self.norm(self.conv_3x3(x), arch_ids)
        x = self.tran_1(self.stage_1(x, arch_ids), arch_ids)
        x = self.tran_2(self.stage_2(x, arch_ids), arch_ids)
        if self.training and self.aux:
            logits_aux = self.auxiliary_head(x)
            x = self.stage_3(x, arch_ids)
            return self.classifier(x), logits_aux
        else:
            x = self.stage_3(x, arch_ids)
            return self.classifier(x)

def wrn(args, num_classes=10):
    model = WRN(args, num_classes)
    return model
