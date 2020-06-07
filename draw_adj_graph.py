# -*- coding: utf-8 -*-
import sys
from graphviz import Digraph
import torch
import math
import re
import numpy as np

colores_list = ["yellow3", "palevioletred3", "cyan3", "grey90"]
# adj_path = "/data/zhijie/snapshots_fbnn/random-p0.6-7/log_seed_1.txt" #
# adjs = re.findall('(array\(.*?\))', ''.join(open(adj_path).readlines()), flags=re.IGNORECASE|re.MULTILINE|re.DOTALL)
# adjs = [np.array(eval(adj[6:-1])) for adj in adjs]
# adjs = [np.array([[1, 0, 0, 0],
#                   [0, 1, 0, 0],
#                   [0, 1, 1, 0],
#                   [1, 0, 1, 1],])]
np.random.seed(1)
seed = 5
num_nodes = 8
arch_p = 0.7
rng = np.random.RandomState(seed)
adjs = []
for j in range(3):
    adj_one = np.tril(rng.rand(num_nodes+1, num_nodes+1) > arch_p, -1).astype(np.int) + np.eye(num_nodes+1).astype(np.int)
    adjs.append(adj_one)
print(adjs[0])

total_len = sum([adjs[i].shape[0] for i in range(len(adjs))]) + 1
whole_adj = np.zeros((total_len, total_len))
start = 1
starts, ends = [], []
for i in range(len(adjs)):
    starts.append(start-1)
    whole_adj[start: start+adjs[i].shape[0], start-1: start-1+adjs[i].shape[1]] = adjs[i]
    start += adjs[i].shape[0]
ends.append(start-1)


g = Digraph(
  format='pdf',
  edge_attr=dict(fontsize='20', fontname="times"),
  node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='1.5', penwidth='2', fontname="times"),
  engine='dot')
g.body.extend(['rankdir=LR'])

# with g.subgraph(name='child', node_attr={'shape': 'box', 'height': '0.01', 'style': 'invisible'}) as c:
#     if dataset == "cifar10" or dataset == "cifar100":
#         c.edge('none', 'foo0', style="invisible", fillcolor="white", color="white", label=" ")
#         c.edge('foo0', 'bar0', fillcolor="grey90", label="conv_1×1",  penwidth=str(1))
#     else:
#         c.edge('foo0', 'bar0', fillcolor="grey90", label="conv_3×3",  penwidth=str(1))
#     c.edge('bar0', 'bar1', color=colores_list[0], label="skip_connect",  penwidth=str(1))
#     c.edge('bar1', 'bar2', color=colores_list[1], label="sep_conv_3×3", penwidth=str(1))
#     c.edge('bar2', 'bar3', color=colores_list[2], label="dil_conv_3×3", penwidth=str(1))

for i in range(whole_adj.shape[0]):
    if i in starts:
        fillcolor = 'darkseagreen2'
        if i == 0:
            label = 'pre-process'
        else:
            label = 'pooling'
    elif i in ends:
        fillcolor = 'palevioletred3'
        label = 'pooling'
    else:
        fillcolor = 'lightblue'
        label = 'conv3×3'
    g.node(str(i), fillcolor=fillcolor, label=label)
    for j in range(i):
        if whole_adj[i, j] == 1:
            g.edge(str(j), str(i), fillcolor="cyan3", penwidth=str(1))


g.render('pdfs/' + str(seed), view=False)