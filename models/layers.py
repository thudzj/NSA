import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def orthogonal_init(param, num_cats, num_sharing):
    d_inits = torch.zeros((num_cats,num_sharing))
    nn.init.orthogonal_(d_inits)
    for j in range(num_cats):
        param.weights[j].data = d_inits[j]

def sparse_init(param, num_cats, num_sharing):
    d_inits = torch.zeros((num_sharing,num_cats))
    nn.init.sparse_(d_inits, 0.5)
    d_inits = d_inits.transpose(0,1)
    d_inits = d_inits.abs()
    d_inits /= d_inits.sum(1, keepdim=True)
    for j in range(num_cats):
        param.weights[j].data = d_inits[j]

def identity_init(param, num_cats, num_sharing):
    if num_sharing == 1:
        coeffs = torch.ones((num_cats, 1))
    else:
        coeffs = torch.zeros((num_cats, num_sharing))
        ratio = ( num_sharing -1.)/(num_cats-1.)
        coeffs[0,0] = 1
        i = 1
        for k in range(1,num_cats-1):
            i += ratio
            low = int(i) - 1
            w2 = i - int(i)
            w1 = 1 - w2
            coeffs[k,low] = w1
            coeffs[k,low+1] = w2
        coeffs[-1,-1] = 1
    for j in range(num_cats):
        param.weights[j].data = coeffs[j]

class ArchConditionalWeight(nn.Module):
    def __init__(self, num_archs, shape, construct_fn=torch.zeros):
        super(ArchConditionalWeight, self).__init__()
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(construct_fn(shape)) for _ in range(num_archs)])

    def forward(self, arch_id):
        return self.weights[arch_id]

class TConv2d(nn.Module):
    def __init__(self, num_archs, inplanes, outplanes, kernel_size=3, stride=1, padding=1):
        super(TConv2d, self).__init__()
        self.num_archs = num_archs
        self.stride = stride
        self.padding = padding
        kernel = torch.Tensor(outplanes, inplanes, kernel_size, kernel_size)
        init.kaiming_normal_(kernel)
        self.kernel = nn.Parameter(kernel)
        kernel2 = torch.Tensor(outplanes, inplanes, kernel_size, kernel_size)
        init.kaiming_normal_(kernel2)
        self.kernel2 = nn.Parameter(kernel2)
        self.adaptive_weights = ArchConditionalWeight(num_archs, (outplanes, 1, 1, 1), torch.zeros)

        aw_inits = torch.zeros((num_archs,outplanes,1,1,1))
        nn.init.orthogonal_(aw_inits)
        for i in range(num_archs):
            self.adaptive_weights.weights[i].data = aw_inits[i]

    def forward(self, input, arch_id):
        kernel = self.adaptive_weights(arch_id)*self.kernel2 + self.kernel
        return F.conv2d(input, kernel, stride=self.stride, padding=self.padding)

class TLinear(nn.Module):
    def __init__(self, num_archs, in_features, out_features, bias=True):
        super(TLinear, self).__init__()
        self.num_archs = num_archs
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_arch = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.bias_arch = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_arch', None)
        self.reset_parameters()

        self.adaptive_weights = ArchConditionalWeight(num_archs, (out_features, 1), torch.zeros)
        self.adaptive_weights_1 = ArchConditionalWeight(num_archs, (out_features,), torch.zeros)

        aw_inits1 = torch.zeros((num_archs,out_features,1))
        nn.init.orthogonal_(aw_inits1)
        aw_inits2 = torch.zeros((num_archs,out_features))
        nn.init.orthogonal_(aw_inits2)
        for i in range(num_archs):
            self.adaptive_weights.weights[i].data = aw_inits1[i]
            self.adaptive_weights_1.weights[i].data = aw_inits2[i]

    def reset_parameters(self):
        init.kaiming_normal_(self.weight)
        init.kaiming_normal_(self.weight_arch)
        if self.bias is not None:
            self.bias.data.zero_()
            self.bias_arch.data.zero_()

    def forward(self, input, arch_id):
        return F.linear(input, self.weight+self.weight_arch*self.adaptive_weights(arch_id), self.bias+self.bias_arch*self.adaptive_weights_1(arch_id))


class CConv2d(nn.Module):
    def __init__(self, num_cats, num_sharing, init_type, inplanes, outplanes, kernel_size=3, stride=1, padding=1):
        super(CConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernels = torch.nn.Parameter(torch.zeros(num_sharing, outplanes, inplanes, kernel_size, kernel_size))

        for i in range(num_sharing):
            init.kaiming_normal_(self.kernels[i])

        self.adaptive_weights = ArchConditionalWeight(num_cats, (num_sharing,), torch.zeros)
        if init_type == 'orth':
            orthogonal_init(self.adaptive_weights, num_cats, num_sharing)
        elif init_type == 'idt':
            identity_init(self.adaptive_weights, num_cats, num_sharing)
        elif init_type == 'spr':
            sparse_init(self.adaptive_weights, num_cats, num_sharing)
        else:
            raise Error

    def forward(self, input, arch_id):
        return F.conv2d(input, (self.adaptive_weights(arch_id).view(-1, 1, 1, 1, 1)*self.kernels).sum(0), stride=self.stride, padding=self.padding)

class CLinear(nn.Module):
    def __init__(self, num_cats, num_sharing, init_type, in_features, out_features, bias=True):
        super(CLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.nn.Parameter(torch.zeros(num_sharing, out_features, in_features))
        if bias:
            self.biases = torch.nn.Parameter(torch.zeros(num_sharing, out_features))
        else:
            self.register_parameter('biases', None)
        
        for i in range(num_sharing):
            init.kaiming_normal_(self.weights[i])
            if self.biases is not None:
                self.biases[i].data.zero_()

        self.adaptive_weights = ArchConditionalWeight(num_cats, (num_sharing,), torch.zeros)
        self.adaptive_weights_1 = ArchConditionalWeight(num_cats, (num_sharing,), torch.zeros)

        if init_type == 'orth':
            orthogonal_init(self.adaptive_weights, num_cats, num_sharing)
            orthogonal_init(self.adaptive_weights_1, num_cats, num_sharing)
        elif init_type == 'idt':
            identity_init(self.adaptive_weights, num_cats, num_sharing)
            identity_init(self.adaptive_weights_1, num_cats, num_sharing)
        elif init_type == 'spr':
            sparse_init(self.adaptive_weights, num_cats, num_sharing)
            sparse_init(self.adaptive_weights_1, num_cats, num_sharing)
        else:
            raise Error


    def forward(self, input, cat):
        return F.linear(input, (self.adaptive_weights(cat).view(-1,1,1)*self.weights).sum(0), (self.adaptive_weights_1(cat).view(-1,1)*self.biases).sum(0) if self.biases is not None else None)

class TemplateBank(nn.Module):
    def __init__(self, num_templates, in_planes, out_planes, kernel_size):
        super(TemplateBank, self).__init__()
        self.coefficient_shape = (num_templates,1,1,1,1)
        templates = [torch.Tensor(out_planes, in_planes, kernel_size, kernel_size) for _ in range(num_templates)]
        for i in range(num_templates): init.kaiming_normal_(templates[i])
        self.templates = nn.Parameter(torch.stack(templates))

    def forward(self, coefficients):
        return (self.templates*coefficients).sum(0)

class SConv2d(nn.Module):
    def __init__(self, bank, num_archs, stride=1, padding=1):
        super(SConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.bank = bank
        self.coefficients = ArchConditionalWeight(num_archs, bank.coefficient_shape)

    def forward(self, input, arch_id):
        params = self.bank(self.coefficients(arch_id))
        return F.conv2d(input, params, stride=self.stride, padding=self.padding)

class SBN2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 track_running_stats=True):
        super(SBN2d, self).__init__(num_features, eps, momentum, False, track_running_stats)

    def forward(self, input, weight, bias):
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, weight, bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

class ArchConditionalConv1x1(torch.nn.Module):
    def __init__(self, C_in, num_cats=10):
        super(ArchConditionalConv1x1, self).__init__()
        self.convs = nn.ModuleList([torch.nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=True, groups=C_in) for  _ in range(num_cats)])
        self.num_cats = num_cats

    def forward(self, input, cat):
        return self.convs[cat](input)


class ArchConditionalBatchNorm(torch.nn.Module):
    def __init__(self, num_features, num_cats=10, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(ArchConditionalBatchNorm, self).__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_cats, num_features))
            self.register_buffer('running_var', torch.ones(num_cats, num_features))
            self.register_buffer('num_batches_tracked', torch.zeros(num_cats, dtype=torch.long))
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

    def forward(self, input, cat):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked[cat] += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked[cat].item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean[cat], self.running_var[cat], self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, num_cats={num_cats}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

class TBatchNorm(torch.nn.Module):
    def __init__(self, num_features, num_cats=10, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(TBatchNorm, self).__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_features))
            self.weight_arch = torch.nn.Parameter(torch.Tensor(num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_features))
            self.bias_arch = torch.nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('weight_arch', None)
            self.register_parameter('bias', None)
            self.register_parameter('bias_arch', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_cats, num_features))
            self.register_buffer('running_var', torch.ones(num_cats, num_features))
            self.register_buffer('num_batches_tracked', torch.zeros(num_cats, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

        self.adaptive_weights = ArchConditionalWeight(num_cats, (num_features,), torch.zeros)
        self.adaptive_weights_1 = ArchConditionalWeight(num_cats, (num_features,), torch.zeros)

        aw_inits1 = torch.zeros((num_cats,num_features))
        nn.init.orthogonal_(aw_inits1)
        aw_inits2 = torch.zeros((num_cats,num_features))
        nn.init.orthogonal_(aw_inits2)
        for i in range(num_cats):
            self.adaptive_weights.weights[i].data = aw_inits1[i]
            self.adaptive_weights_1.weights[i].data = aw_inits2[i]

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.weight_arch.data.fill_(1.0)
            self.bias.data.zero_()
            self.bias_arch.data.zero_()

    def forward(self, input, cat):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked[cat] += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked[cat].item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean[cat], self.running_var[cat], self.weight+self.weight_arch*self.adaptive_weights(cat), self.bias+self.bias_arch*self.adaptive_weights_1(cat),
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)


class FullArchConditionalBatchNormN(torch.nn.Module):
    def __init__(self, num_features, num_cats=10, num_sharing=10, init_type='orth', eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(FullArchConditionalBatchNormN, self).__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weights = torch.nn.Parameter(torch.Tensor(num_sharing, num_features))
            self.biases = torch.nn.Parameter(torch.Tensor(num_sharing, num_features))
        else:
            self.register_parameter('weights', None)
            self.register_parameter('biases', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_cats, num_features))
            self.register_buffer('running_var', torch.ones(num_cats, num_features))
            self.register_buffer('num_batches_tracked', torch.zeros(num_cats, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

        self.adaptive_weights = ArchConditionalWeight(num_cats, (num_sharing,), torch.zeros)
        self.adaptive_weights_1 = ArchConditionalWeight(num_cats, (num_sharing,), torch.zeros)

        if init_type == 'orth':
            orthogonal_init(self.adaptive_weights, num_cats, num_sharing)
            orthogonal_init(self.adaptive_weights_1, num_cats, num_sharing)
        elif init_type == 'idt':
            identity_init(self.adaptive_weights, num_cats, num_sharing)
            identity_init(self.adaptive_weights_1, num_cats, num_sharing)
        elif init_type == 'spr':
            sparse_init(self.adaptive_weights, num_cats, num_sharing)
            sparse_init(self.adaptive_weights_1, num_cats, num_sharing)
        else:
            raise Error

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weights.data.fill_(1.0)
            self.biases.data.zero_()

    def forward(self, input, cat):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked[cat] += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked[cat].item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean[cat], self.running_var[cat], (self.adaptive_weights(cat).view(-1,1)*self.weights).sum(0), (self.adaptive_weights_1(cat).view(-1,1)*self.biases).sum(0),
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, num_cats={num_cats}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

class FullArchConditionalBatchNorm(torch.nn.Module):
    def __init__(self, num_features, num_cats=10, affine=True,
                 track_running_stats=True):
        super(FullArchConditionalBatchNorm, self).__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm2d(num_features, affine=affine, track_running_stats=track_running_stats) for _ in range(num_cats)])

    def forward(self, input, cat):
        return self.bns[cat](input)


class ArchConditionalGroupNorm(torch.nn.Module):
    
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']

    def __init__(self, num_groups, num_channels, num_cats=10, eps=1e-5, affine=True):
        super(ArchConditionalGroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.num_cats = num_cats
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(num_channels)) for _ in range(num_cats)])
            self.biases = torch.nn.ParameterList([torch.nn.Parameter(torch.Tensor(num_channels)) for _ in range(num_cats)])
        else:
            self.register_parameter('weights', None)
            self.register_parameter('biases', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            for param in self.weights:
                init.ones_(param)
            for param in self.biases:
                init.zeros_(param)

    def forward(self, input, arch_id):
        return F.group_norm(
            input, self.num_groups, self.weights[arch_id], self.biases[arch_id], self.eps)

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)