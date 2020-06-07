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

class NormFunc(nn.Module):
    def __init__(self, args, num_features, arch_nums=1):
        super(NormFunc, self).__init__()
        self.arch_nums = arch_nums
        if arch_nums == 1:
            self.fn = nn.BatchNorm2d(num_features,affine=args.affine,track_running_stats=args.track_running_stats)
        else:
            self.fn = ArchConditionalBatchNorm(num_features, arch_nums, affine=args.affine,track_running_stats=args.track_running_stats)
            #ArchConditionalGroupNorm(32, num_features, arch_nums, affine=args.affine)

    def forward(self, x, arch_ids=None):
        if self.arch_nums == 1:
            return self.fn(x)
        else:
            return self.fn(x, arch_ids)

class BNReLuConv(nn.Module):
  def __init__(self, args, inplanes, outplanes, kernel_size=3, stride=1, padding=1, arch_nums=1):
    super(BNReLuConv, self).__init__()
    self.norm = NormFunc(args, inplanes, arch_nums)
    self.relu = nn.ReLU(inplace=True)
    self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    self.dropout = nn.Dropout(p=args.dropout_rate)

  def forward(self, x, arch_ids=None):
    return self.dropout(self.conv(self.relu(self.norm(x, arch_ids))))

class Stage(nn.Module):
  def __init__(self, args, inplanes, adjacencies, outplanes=None):
    super(Stage, self).__init__()
    self.num_nodes = args.num_nodes
    self.dpr = args.droppath_rate
    self.learn_aggr = args.learn_aggr
    self.adj = nn.Parameter(torch.from_numpy(np.stack(adjacencies)), requires_grad=False)
    if self.learn_aggr:
        self.mean_weight = nn.Parameter(torch.ones(self.adj.shape[0], self.num_nodes+1, self.num_nodes+1))
        self.sigmoid = nn.Sigmoid()
    
    self.nodeop = nn.ModuleList()
    for _ in range(self.num_nodes):
        self.nodeop.append(BNReLuConv(args, inplanes, inplanes, arch_nums=self.adj.shape[0]))

    self.norm = NormFunc(args, inplanes, arch_nums=self.adj.shape[0])
    if outplanes is None:
        self.transition = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(8))
    else:
        self.transition = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False), nn.AvgPool2d(kernel_size=2, stride=2))

  def forward(self, x, arch_ids):
    if self.learn_aggr:
        mask = (self.adj * self.sigmoid(self.mean_weight))[arch_ids, :, :]
    else:
        mask = self.adj[arch_ids, :, :]

    outputs = [x]
    for i in range(self.num_nodes):
        outputs.append(self.nodeop[i](sum(outputs[j]*mask[:, i:(i+1), j:(j+1),None].contiguous() for j in range(i+1)),
                                      arch_ids))
    return self.transition(self.norm(sum(outputs[j]*mask[:, -1:, j:(j+1), None].contiguous() for j in range(self.num_nodes+1)), arch_ids))

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
        self.stage_1 = Stage(args, n_channels[0], adjacencies[0], n_channels[1])
        self.stage_2 = Stage(args, n_channels[1], adjacencies[1], n_channels[2])
        self.stage_3 = Stage(args, n_channels[2], adjacencies[2])
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(n_channels[2], num_classes))

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
        x = self.conv_3x3(x)
        x = self.stage_1(x, arch_ids)
        x = self.stage_2(x, arch_ids)
        x = self.stage_3(x, arch_ids)
        return self.classifier(x)

def wrn(args, num_classes=10):
    model = WRN(args, num_classes)
    return model