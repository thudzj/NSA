import torch
import torch.nn as nn
from torch.nn import init
from .layers import *
from functools import partial
import numpy as np

class NormFunc(nn.Module):
    def __init__(self, args, num_features):
        super(NormFunc, self).__init__()
        self.fn = nn.BatchNorm2d(num_features,affine=args.affine,track_running_stats=args.track_running_stats)

    def forward(self, x):
        return self.fn(x)

class BNReLuConv(nn.Module):
  def __init__(self, args, inplanes, outplanes, kernel_size=3, stride=1, padding=1):
    super(BNReLuConv, self).__init__()
    self.norm = NormFunc(args, inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    self.dropout = nn.Dropout(p=args.dropout_rate)

  def forward(self, x):
    return self.dropout(self.conv(self.relu(self.norm(x))))

class Transition(nn.Module):
    def __init__(self, args, inplanes, outplanes):
        super(Transition, self).__init__()
        self.node = BNReLuConv(args, inplanes, outplanes, 1, 1, 0)
        
    def forward(self, x):
        return F.avg_pool2d(self.node(x), 2)

class Stage(nn.Module):
  def __init__(self, args, planes):
    super(Stage, self).__init__()
    self.num_nodes = args.num_nodes
    self.dpr = args.droppath_rate
    self.learn_aggr = args.learn_aggr
    if self.learn_aggr:
        self.mean_weight = nn.Parameter(torch.ones(self.num_nodes+1, self.num_nodes+1))
        self.sigmoid = nn.Sigmoid()
    
    self.nodeop = nn.ModuleList()
    for _ in range(self.num_nodes):
        self.nodeop.append(BNReLuConv(args, planes, planes))

  def drop_path(self, x, i, j):
    if self.dpr > 0. and i != j:
        keep_prob = 1.-self.dpr
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x = x.div(keep_prob).mul_(mask)
    if self.learn_aggr:
        return self.sigmoid(self.mean_weight[i, j]) * x
    else:
        return x

  def forward(self, x, adj):
    outputs = [x]
    for i in range(self.num_nodes):
        outputs.append(self.nodeop[i](sum(self.drop_path(outputs[j], i, j) for j in range(i+1) if adj[i, j] == 1)))
    return sum(self.drop_path(outputs[j], self.num_nodes, j) for j, v in enumerate(adj[-1]) if v == 1)

class WRN(nn.Module):
    def __init__(self, args, num_classes=10):
        super(WRN, self).__init__()
        n_channels = [16*args.width, 32*args.width, 64*args.width]
        self.num_classes = num_classes

        self.conv_3x3 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.stage_1 = Stage(args, n_channels[0])
        self.tran_1 = Transition(args, n_channels[0], n_channels[1])

        self.stage_2 = Stage(args, n_channels[1])
        self.tran_2 = Transition(args, n_channels[1], n_channels[2])

        self.stage_3 = Stage(args, n_channels[2])
        self.lastact = nn.Sequential(NormFunc(args,n_channels[2]),nn.ReLU(inplace=True),nn.AvgPool2d(8),nn.Flatten())

        self.classifier = nn.Linear(n_channels[2], num_classes)

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

    def forward(self, x, adjacencies):
        x = self.conv_3x3(x)
        x = self.tran_1(self.stage_1(x, adjacencies[0]))
        x = self.tran_2(self.stage_2(x, adjacencies[1]))
        x = self.lastact(self.stage_3(x, adjacencies[2]))
        return self.classifier(x)

def wrn(args, num_classes=10):
    model = WRN(args, num_classes)
    return model