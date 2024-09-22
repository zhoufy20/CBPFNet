# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:04:06 2023

@author: ZHANG Jun
"""

import torch
import torch.nn as nn
from dgl import function as fn
from dgl.ops import edge_softmax

class Layer(nn.Module):
    """ Single graph attention network for crystal. """
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_heads,
                 device='cuda',
                 bias=True,
                 negative_slope=0.2):
        super(Layer, self).__init__()
        # input args
        self.in_dim         = in_dim
        self.out_dim        = out_dim
        self.num_heads      = num_heads
        self.device         = device
        self.bias           = bias
        self.negative_slope = negative_slope

        # initialize trainable parameters,
        # w_att_src : torch.Size([1, self.num_heads, self.out_dim])
        self.w_att_src = nn.Parameter(torch.randn(1, self.num_heads, self.out_dim, device=self.device))
        self.w_att_dst = nn.Parameter(torch.randn(1, self.num_heads, self.out_dim, device=self.device))

        # dense layers
        self.layer = nn.Linear(self.in_dim, self.out_dim*self.num_heads, bias=self.bias, device=self.device)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=self.negative_slope)

        # leaky relu function
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=self.negative_slope)

    def forward(self, feat, dist, graph): # feat with shape of (number of nodes, number of features of each node). The graph can have no values of nodes.
        """ Forward this `GAT` layer, and feat is the input feature of all nodes."""

        # h = self.gat_layers[l](h, dist, graph)
        # dist : (number of edges, number of heads, 1)
        # feat_src : (number of nodes, out_dim*num_heads)
        feat_src = feat_dst = self.leaky_relu1(self.layer(feat))

        # feat_src : (number of nodes, number of heads, num out)
        feat_src = torch.reshape(feat_src, (-1, self.num_heads, self.out_dim))
        feat_dst = torch.reshape(feat_dst, (-1, self.num_heads, self.out_dim))

        # feat_src:(number of nodes, number of heads, num out)
        # w_att_src: (1, self.num_heads, self.out_dim)
        # A broadcasting for w_att_src, and
        e_src = torch.sum(feat_src * self.w_att_src, axis=-1, keepdim=True) # shape of e_src: (number of nodes, number of heads, 1)
        e_dst = torch.sum(feat_dst * self.w_att_dst, axis=-1, keepdim=True) # shape of e_dst: (number of nodes, number of heads, 1)

        # save on nodes
        graph.srcdata.update({'ft': feat_src, 'e_src': e_src})
        graph.dstdata.update({'e_dst': e_dst})

        # save on edges
        graph.apply_edges(fn.u_add_v('e_src', 'e_dst', 'e'))   # shape of e: (number of edges, number of heads, 1) # similar to the original paper, but feed to dense first, summation is the second.
        e = self.leaky_relu2(graph.edata.pop('e'))  # shape of e: (number of edges, number of heads, 1)

        dist = torch.reshape(dist, (-1, self.num_heads, 1))  # shape of dist : (number of edges, number of heads, 1)

        # the attention scores of every edge
        graph.edata['a']  = edge_softmax(graph, e) * dist # shape of a: (number of edges, number of heads, 1)

        graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft')) # shape of ft: (number of nodes, number of heads, number of out)
        dst = torch.reshape(graph.ndata['ft'], (-1, self.num_heads * self.out_dim)) # shape of `en`: (number of nodes, number of heads * number of out)
        return dst # node_energy, node_force

if __name__ == '__main__':
    import dgl
    pass
