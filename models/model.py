#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :atomic_feature.py
# @Time      :2024/3/24 13:51
# @Author    :Feiyu
# @Main      :The framework of agat.
#===================================================================================================
#### reference from DOI:https://doi.org/10.1016/j.joule.2023.06.003
#### reference code from url:https://github.com/jzhang-github/AGAT
###  cite:Zhang, Jun & Wang, Chaohui & Huang, Shasha & Xiang, Xuepeng & Xiong, Yaoxu & Biao, Xu
###  & Ma, Shihua & Fu, Haijun & Kai, Jijung & Kang, Xiongwu & Zhao, Shijun. (2023).
###  Design high-entropy electrocatalyst via interpretable deep graph attention learning.
###  Joule. 7. 10.1016/j.joule.2023.06.003.
#===================================================================================================


import os
import json
import torch
import torch.nn as nn

from dgl.ops import edge_softmax
from dgl import function as fn
from dgl.data.utils import load_graphs

from models.layer import Layer

class PotentialModel(nn.Module):
    """A GAT models with multiple gat layers for predicting atomic energies and forces tensors."""

    """".. Note:: You can also use this models to train and predict atom and bond related properties.
    You need to store the labels on graph edges if you want to do so. This models has multiple attention heads.

    .. Important::

        The first value of ``gat_node_dim_list`` is the depth of atomic representation.

        The first value of ``energy_readout_node_list``, ``force_readout_node_list``, ``stress_readout_node_list`` is
        the input dimension and equals to last value of `gat_node_list * num_heads`.

        The last values of ``energy_readout_node_list``, ``force_readout_node_list``, ``stress_readout_node_list`` are 
        ``1``, ``3``, and ``6``, respectively.
    """
    def __init__(self,
                 gat_node_dim_list,
                 energy_readout_node_list,
                 force_readout_node_list,
                 head_list=['div'],
                 bias=True,
                 negative_slope=0.2,
                 device = 'cuda',
                 tail_readout_no_act=[3,3]):
        super(PotentialModel, self).__init__()

        self.gat_node_dim_list = gat_node_dim_list
        self.energy_readout_node_list = energy_readout_node_list
        self.force_readout_node_list = force_readout_node_list
        self.head_list = head_list
        self.bias = bias
        self.device = device
        self.negative_slope = negative_slope
        self.tail_readout_no_act = tail_readout_no_act

        self.num_gat_layers = len(self.gat_node_dim_list)-1
        self.num_energy_readout_layers = len(self.energy_readout_node_list)-1
        self.num_force_readout_layers = len(self.force_readout_node_list)-1
        self.num_heads = len(self.head_list)

        self.__gat_real_node_dim_list = [x*self.num_heads for x in self.gat_node_dim_list[1:]]
        self.__gat_real_node_dim_list.insert(0,self.gat_node_dim_list[0])
        # self.energy_readout_node_list.insert(0, self.__gat_real_node_dim_list[-1]) # calculate and insert input dimension.
        # self.force_readout_node_list.insert(0, self.__gat_real_node_dim_list[-1])
        # self.stress_readout_node_list.insert(0, self.__gat_real_node_dim_list[-1])

        # register layers and parameters.
        self.gat_layers = nn.ModuleList([])
        self.energy_readout_layers = nn.ModuleList([])
        self.force_readout_layers = nn.ModuleList([])

        # gat layers(agat.Layer)
        # 'gat_node_dim_list': [len(default_elements), 10, 20, 50, 100, 100],
        # '__gat_real_node_dim_list':[len(default_elements), 60, 120, 300, 600, 600]
        # 'self.gat_layers':[len(default_elements), 10*6, 20*6, 50*6, 100*6, 100*6]
        for l in range(self.num_gat_layers):
            self.gat_layers.append(Layer(self.__gat_real_node_dim_list[l],
                                            self.gat_node_dim_list[l+1],
                                            self.num_heads,
                                            device=self.device,
                                            bias=self.bias,
                                            negative_slope=self.negative_slope))
        # energy readout layer(Linear)
        # 'energy_readout_node_list': [600, 500, 400, 300, 150, 100, 50, 25, 10, 5, FIX_VALUE[0]],
        # 'tail_readout_no_act' : [3,3]
        for l in range(self.num_energy_readout_layers-self.tail_readout_no_act[0]):
            self.energy_readout_layers.append(nn.Linear(self.energy_readout_node_list[l],
                                                         self.energy_readout_node_list[l+1],
                                                         self.bias, self.device))
            self.energy_readout_layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
        for l in range(self.tail_readout_no_act[0]):
            self.energy_readout_layers.append(nn.Linear(self.energy_readout_node_list[l-self.tail_readout_no_act[0]-1],
                                                         self.energy_readout_node_list[l-self.tail_readout_no_act[0]],
                                                         self.bias, self.device))

        # force readout layer
        # 'force_readout_node_list': [600, 500, 400, 300, 150, 100, 50, 25, 10, 5, FIX_VALUE[1]],
        for l in range(self.num_force_readout_layers-self.tail_readout_no_act[1]):
            self.force_readout_layers.append(nn.Linear(self.force_readout_node_list[l],
                                                        self.force_readout_node_list[l+1],
                                                        self.bias, self.device))
            self.force_readout_layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
        for l in range(self.tail_readout_no_act[1]):
            self.force_readout_layers.append(nn.Linear(self.force_readout_node_list[l-self.tail_readout_no_act[1]-1],
                                                        self.force_readout_node_list[l-self.tail_readout_no_act[1]],
                                                        self.bias, self.device))
        # # stress readout layer
        # for l in range(self.num_stress_readout_layers-self.tail_readout_no_act[2]):
        #     self.stress_readout_layers.append(nn.Linear(self.stress_readout_node_list[l],
        #                                                  self.stress_readout_node_list[l+1],
        #                                                  self.bias, self.device))
        #     self.stress_readout_layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
        # for l in range(self.tail_readout_no_act[2]):
        #     self.stress_readout_layers.append(nn.Linear(self.stress_readout_node_list[l-self.tail_readout_no_act[2]-1],
        #                                                  self.stress_readout_node_list[l-self.tail_readout_no_act[2]],
        #                                                  self.bias, self.device))
        #
        # self.u2e = nn.Linear(self.gat_node_dim_list[0], self.stress_readout_node_list[0],
        #                      False, self.device) # propogate source nodes to edges.

        self.__real_num_energy_readout_layers = len(self.energy_readout_layers)
        self.__real_num_force_readout_layers = len(self.force_readout_layers)

        # attention heads
        self.head_fn = {'mul' : self.mul,
                        'div' : self.div,
                        'free': self.free,
                        'sigmoid':self.sigmoid,
                        'softmax':self.softmax,
                        'leaky_relu':self.leaky_relu}

    def mul(self, TorchTensor):
        """ Multiply head. """

        return TorchTensor

    def div(self, TorchTensor):
        """ Division head. """

        return 1/TorchTensor

    def free(self, TorchTensor):
        """ Free head. """
        return torch.ones(TorchTensor.size(), device=self.device)

    def sigmoid(self, TorchTensor):
        """ Sigmoid attention head. """
        # Apply Sigmoid function element-wise to the input tensor
        return torch.sigmoid(TorchTensor)

    def softmax(self, TorchTensor):
        """ Softmax attention head. """
        # Apply Softmax function along the last dimension of the input tensor
        return torch.softmax(TorchTensor, dim=-1)

    def leaky_relu(self, TorchTensor, negative_slope=0.01):
        """ Leaky ReLU attention head. """
        # Apply Leaky ReLU function element-wise to the input tensor
        return (TorchTensor > 0) * TorchTensor + (TorchTensor <= 0) * (negative_slope * TorchTensor)

    def get_head_mechanism(self, fn_list, TorchTensor):
        """ Get attention heads
        :param fn_list: A list of head mechanisms. For example: ['mul', 'div', 'free']
        :param TorchTensor: A PyTorch tensor
        :return: A new tensor after the transformation. """
        TorchTensor_list = []
        for func in fn_list:
            TorchTensor_list.append(self.head_fn[func](TorchTensor))
        return torch.cat(TorchTensor_list, 1)

    def forward(self, graph):
        """The `forward` function of PotentialModel models, and return tuple about energy and force. """
        with graph.local_scope():
            # h : (number of nodes, dimension of one-hot code representation)
            h    = graph.ndata['h']

            # graph.edata['dist'] : (number of edges)
            # dist : (number of edges, 1, 1)
            dist = torch.reshape(graph.edata['dist'], (-1, 1, 1))
            dist = torch.where(dist < 0.01, 0.01, dist)

            # shape of dist: (number of edges, number of heads, 1)
            dist = self.get_head_mechanism(self.head_list, dist)

            # 'self.gat_layers':[len(default_elements), 10*6, 20*6, 50*6, 100*6, 100*6]
            for l in range(self.num_gat_layers):
                h = self.gat_layers[l](h, dist, graph)                 # shape of h: (number of nodes, number of heads * num_out)

            # predict energy
            energy = h
            for l in range(self.__real_num_energy_readout_layers):
                energy = self.energy_readout_layers[l](energy)
            batch_nodes = graph.batch_num_nodes().tolist()
            energy = torch.split(energy, batch_nodes)
            energy = torch.stack([torch.mean(e) for e in energy])

            # Predict force
            graph.ndata['node_force'] = h
            graph.apply_edges(fn.u_add_v('node_force', 'node_force', 'force_score'))   # shape of score: (number of edges, ***, 1)
            force_score = torch.reshape(graph.edata['force_score'],(-1, self.num_heads, self.gat_node_dim_list[-1])) / dist
            force_score = torch.reshape(force_score, (-1, self.num_heads * self.gat_node_dim_list[-1]))

            for l in range(self.__real_num_force_readout_layers):
                force_score = self.force_readout_layers[l](force_score)
            graph.edata['force_score_vector'] = force_score * graph.edata['direction']      # shape (number of edges, 1)
            graph.update_all(fn.copy_e('force_score_vector', 'm'), fn.sum('m', 'force_pred'))        # shape of graph.ndata['force_pred']: (number of nodes, 3)
            force = graph.ndata['force_pred']

            return energy, force

class CrystalPropertyModel(nn.Module):
    pass

class AtomicPropertyModel(nn.Module):
    pass

class AtomicVectorModel(nn.Module):
    pass

if __name__ == '__main__':
    import dgl
    g_list, l_list = dgl.load_graphs('../dataset/C2CTestset/all_graphs.bin')
    head_list= ['mul', 'div', 'free', 'sigmoid', 'softmax', 'leaky_relu']
    gat_node_dim_list= [4, 10, 20, 50, 100, 100]
    energy_readout_node_list = [600, 500, 400, 300, 150, 100, 50, 25, 10, 5, 3] # the first value should be: len(head_list)*gat_node_dim_list[-1]
    force_readout_node_list = [600, 500, 400, 300, 150, 100, 50, 25, 10, 5, 1]# the first value should be: len(head_list)*gat_node_dim_list[-1]

    graph = g_list[1]
    feat = graph.ndata['h']
    dist = graph.edata['dist']

    PM = PotentialModel(gat_node_dim_list,
                        energy_readout_node_list,
                        force_readout_node_list,
                        head_list,
                        bias=True,
                        negative_slope=0.2,
                        device = 'cpu',
                        tail_readout_no_act=[3,3]
                        )

    energy, force = PM.forward(graph)

    model = PM
