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

# class SELayer(nn.Module):
#     def __init__(self, channel, arch_nums=1, reduction=16, if_use=False):
#         super(SELayer, self).__init__()
#         self.if_use = if_use
#         if self.if_use:
#             self.arch_nums = arch_nums
#             self.avg_pool = nn.AdaptiveAvgPool2d(1)
#             # self.relu = nn.ReLU(inplace=True)
#             # self.sigmoid = nn.Sigmoid()

#             # tmp = []
#             # for i in range(self.arch_nums):
#             #     weight = torch.Tensor(channel // reduction, channel)
#             #     init.kaiming_uniform_(weight, a=math.sqrt(5))
#             #     # init.kaiming_normal_(weight)
#             #     tmp.append(weight.t())
#             # self.weight1 = nn.Parameter(torch.stack(tmp))

#             # tmp = []
#             # for i in range(self.arch_nums):
#             #     weight = torch.Tensor(channel, channel // reduction)
#             #     init.kaiming_uniform_(weight, a=math.sqrt(5))
#             #     # init.kaiming_normal_(weight)
#             #     tmp.append(weight.t())
#             # self.weight2 = nn.Parameter(torch.stack(tmp))
#             self.fc = nn.Sequential(
#                 nn.Linear(channel, channel // reduction, bias=False),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(channel // reduction, channel, bias=False),
#                 nn.Sigmoid()
#             )

#     def forward(self, x, arch_ids=None):
#         if self.if_use:
#             b, c, _, _ = x.size()
#             # y = self.avg_pool(x).view(b, 1, c)
#             # y = self.relu(y.matmul(self.weight1[arch_ids]))
#             # y = self.sigmoid(y.matmul(self.weight2[arch_ids]))
#             # return x * y.view(b, c, 1, 1).expand_as(x)
#             y = self.avg_pool(x).view(b, c)
#             y = self.fc(y).view(b, c, 1, 1)
#             return x * y.expand_as(x)
#         else:
#             return x

class ReLuConvBN(nn.Module):
  def __init__(self, args, inplanes, outplanes, kernel_size=3, stride=1, padding=1, arch_nums=1):
    super(ReLuConvBN, self).__init__()
    self.relu = nn.ReLU(inplace=True)
    self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    self.dropout = nn.Dropout(p=args.dropout_rate)
    self.norm = NormFunc(args, outplanes, arch_nums)
    # self.se = SELayer(outplanes, arch_nums, if_use=args.se)

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

'''
5-13-2020
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
        self.arch_nums = arch_nums
        if arch_nums == 1:
            self.fn = nn.BatchNorm2d(num_features,affine=args.affine,track_running_stats=args.track_running_stats)
        else:
            self.fn = ArchConditionalBatchNorm(num_features, arch_nums, 
                    affine=args.affine,track_running_stats=args.track_running_stats)

    def forward(self, x, arch_ids=None):
        if self.arch_nums == 1:
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

        self.arch_nums = args.arch_seed_end+1-args.arch_seed_start
        adjacencies = [[], [], []]
        for i in range(args.arch_seed_start, args.arch_seed_end+1):
            rng = np.random.RandomState(i)
            for j in range(3):
                flag = True
                while flag:
                    adj_one = np.tril(rng.rand(args.num_nodes+1, args.num_nodes+1) > args.arch_p, -1).astype(np.int) + np.eye(args.num_nodes+1).astype(np.int)
                    assert(np.all(adj_one.sum(1) > 0))
                    flag = False
                adjacencies[j].append(adj_one)

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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x, _):
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
'''



'''
older
import torch
import torch.nn as nn
from torch.nn import init
from functools import partial
import numpy as np

class ArchConditionalGroupNorm(torch.nn.Module):

    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']

    def __init__(self, num_groups, num_channels, arch_nums, eps=1e-5, affine=True):
        super(ArchConditionalGroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.arch_nums = arch_nums
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(arch_nums, num_channels))
            self.bias = torch.nn.Parameter(torch.Tensor(arch_nums, num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, arch_ids):
        out = torch.nn.functional.group_norm(input, self.num_groups, None, None, self.eps)
        if self.affine:
            shape = [input.size(0), self.num_channels] + (input.dim() - 2) * [1]
            out = torch.addcmul(self.bias[arch_ids].view(shape), self.weight[arch_ids].view(shape), out)
        return out

    def extra_repr(self):
        return '{num_groups}, {num_channels}, {arch_nums}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)


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

def one_hot(y, nb_digits):
    y_onehot = torch.zeros(y.shape[0], nb_digits, device=y.device)
    y_onehot.scatter_(1, y[:, None], 1)
    return y_onehot

class SELayer(nn.Module):
    def __init__(self, channel, arch_nums=1, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.arch_nums = arch_nums
        input_dim = channel
        if self.arch_nums > 1:
            input_dim += self.arch_nums
        self.fc = nn.Sequential(
            nn.Linear(input_dim, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, arch_ids):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        if self.arch_nums > 1:
            y = torch.cat([y, one_hot(arch_ids, self.arch_nums)], 1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class NormFunc(nn.Module):
    def __init__(self, args, num_features, arch_nums=1):
        super(NormFunc, self).__init__()
        self.arch_nums = arch_nums
        if self.arch_nums == 1:
            self.fn = nn.BatchNorm2d(num_features,affine=args.affine,track_running_stats=args.track_running_stats)
        else:
            self.fn = ArchConditionalBatchNorm(num_features, self.arch_nums, affine=args.affine,track_running_stats=args.track_running_stats) # ArchConditionalGroupNorm(32, num_features,s arch_nums, affine=args.affine)

    def forward(self, x, arch_ids=None):
        if self.arch_nums == 1:
            return self.fn(x)
        else:
            return self.fn(x, arch_ids)

class ReLuConvBN(nn.Module):
  def __init__(self, args, inplanes, outplanes, kernel_size=3, stride=1, padding=1, arch_nums=1):
    super(ReLuConvBN, self).__init__()
    self.relu = nn.ReLU(inplace=True)
    self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    self.dropout = nn.Dropout(p=args.dropout_rate)
    self.norm = NormFunc(args, outplanes, 1)
    self.se = args.se
    if self.se:
        self.selayer = SELayer(outplanes, arch_nums)

  def forward(self, x, arch_ids=None):
    output = self.norm(self.dropout(self.conv(self.relu(x))), arch_ids)
    if self.se:
        return self.selayer(output, arch_ids)
    else:
        return output

class Transition(nn.Module):
  def __init__(self, args, inplanes, outplanes, arch_nums=1):
    super(Transition, self).__init__()
    self.relu = nn.ReLU(inplace=True)
    self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    self.norm = nn.BatchNorm2d(outplanes,affine=args.affine,track_running_stats=args.track_running_stats)#NormFunc(args, outplanes, arch_nums) #

  def forward(self, x, arch_ids=None):
    return self.norm(self.pool(self.conv(self.relu(x))))

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

  def __init__(self, args, arch_nums, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.feature_1 = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False))
    self.norm_1 = ArchConditionalBatchNorm(128, arch_nums, affine=args.affine,track_running_stats=args.track_running_stats)
    self.feature_2 = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 768, 2, bias=False))
    self.norm_2 = ArchConditionalBatchNorm(768, arch_nums, affine=args.affine,track_running_stats=args.track_running_stats)
    self.classifier = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Linear(768, num_classes))

  def forward(self, x, arch_ids):
    x = self.feature_1(x)
    x = self.norm_1(x, arch_ids)
    x = self.feature_2(x)
    x = self.norm_2(x, arch_ids)
    x = self.classifier(x.view(x.size(0),-1))
    return x

class WRN(nn.Module):
    def __init__(self, args, num_classes=10):
        super(WRN, self).__init__()
        n_channels = [16*args.width, 32*args.width, 64*args.width]
        self.num_classes = num_classes

        self.arch_nums = args.arch_seed_end+1-args.arch_seed_start
        adjacencies = [[], [], []]
        for i in range(args.arch_seed_start, args.arch_seed_end+1):
            rng = np.random.RandomState(i)
            for j in range(3):
                flag = True
                while flag:
                    adj_one = np.tril(rng.rand(args.num_nodes+1, args.num_nodes+1) > args.arch_p, -1).astype(np.int) + np.eye(args.num_nodes+1).astype(np.int)
                    if np.all(adj_one.sum(1) > 0):
                        flag = False
                adjacencies[j].append(adj_one)

        self.conv_3x3 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(n_channels[0],affine=args.affine,track_running_stats=args.track_running_stats)#NormFunc(args, n_channels[0], self.arch_nums) #
        self.stage_1 = Stage(args, n_channels[0], adjacencies[0])
        self.tran_1 = Transition(args, n_channels[0], n_channels[1], self.arch_nums)
        self.stage_2 = Stage(args, n_channels[1], adjacencies[1])
        self.tran_2 = Transition(args, n_channels[1], n_channels[2], self.arch_nums)
        self.stage_3 = Stage(args, n_channels[2], adjacencies[2])
        self.classifier = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(8), nn.Flatten(), nn.Linear(n_channels[2], num_classes))

        self.aux = args.aux
        if self.aux:
            self.auxiliary_head = AuxiliaryHeadCIFAR(args, self.arch_nums, n_channels[2], num_classes)

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if not 'selayer' in name:
                    init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x, _):
        arch_ids = torch.cuda.LongTensor(x.shape[0]).random_(0, self.arch_nums)
        x = self.norm(self.conv_3x3(x))
        x = self.tran_1(self.stage_1(x, arch_ids), arch_ids)
        x = self.tran_2(self.stage_2(x, arch_ids), arch_ids)
        if self.training and self.aux:
            logits_aux = self.auxiliary_head(x, arch_ids)
            x = self.stage_3(x, arch_ids)
            return self.classifier(x), logits_aux
        else:
            x = self.stage_3(x, arch_ids)
            return self.classifier(x)

def wrn(args, num_classes=10):
    model = WRN(args, num_classes)
    return model
'''