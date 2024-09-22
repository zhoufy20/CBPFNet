# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:54:56 2023

@author: 18326
"""

import torch
from torch.utils.data import Dataset
import dgl
from dgl.data.utils import load_graphs

class LoadDataset(Dataset):
    """ Load the binary graphs. """
    def __init__(self, dataset_path):
        super(LoadDataset, self).__init__()
        self.dataset_path = dataset_path
        self.graph_list, self.props = load_graphs(self.dataset_path) # `props`: properties.

    def __getitem__(self, index):
        """Input the index or slice, return a dgl graph and a dict of Tensor(Graph labels) """

        if isinstance(index, slice):
            graph_list = self.graph_list[index]
            graph = dgl.batch(graph_list)
        else:
            graph = self.graph_list[index]
        props = {k:v[index] for k,v in self.props.items()}
        return graph, props

    def __len__(self):
        """Get the length of the dataset. """
        return len(self.graph_list)


class Collater(object):
    """ The collate function determines how to merge the batch data. """
    def __init__(self, device='cuda'):
        self.device = device

    def __call__(self, data):
        """ Collate the data into batches and return dgl batch graphs, a dict of Tensor(Graph labels) """
        graph_list = [x[0] for x in data]
        graph = dgl.batch(graph_list)

        props = [x[1] for x in data]
        props = {k:torch.stack([x[k] for x in props]) for k in props[0].keys()}
        props = {k:v.to(self.device) for k,v in props.items()}
        return graph.to(self.device), props

if __name__ == '__main__':
    pass