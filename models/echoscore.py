# @Time    : 2024/5/8 21:55
# @Author  : Feiyu
# @File    : Model_Test.py
# @diligent：What doesn't kill me makes me stronger.
# @Function: echo the attention scores.

import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dgl
import networkx as nx
import numpy as np

from models.model import PotentialModel
from lib.model_lib import PearsonR, config_parser, load_state_dict, findordinalatoms
from data.load_dataset import LoadDataset, Collater
from scipy.stats import entropy

from default_parameters import default_train_config
from data.build_dataset import BuildOneGraph
from ase.io import read

class Score(object):
    def __init__(self, **test_config):
        self.test_config = {**default_train_config, **config_parser(test_config)}

        # check device
        self.device = self.test_config['device']
        if torch.cuda.is_available() and self.device == 'cpu':
            print('User warning: `CUDA` device is available, but you choose `cpu`.')
        elif not torch.cuda.is_available() and self.device.split(':')[0] == 'cuda':
            print('User warning: `CUDA` device is not available, but you choose `cuda:0`. Change the device to `cpu`.')
            self.device = 'cpu'
        print('User info: Specified device for potential models:', self.device)

    def echo(self, **test_config):
        # update config if needed.
        self.test_config = {**self.test_config, **config_parser(test_config)}

        model = PotentialModel(self.test_config['gat_node_dim_list'],
                               self.test_config['energy_readout_node_list'],
                               self.test_config['force_readout_node_list'],
                               self.test_config['head_list'],
                               self.test_config['bias'],
                               self.test_config['negative_slope'],
                               self.device,
                               self.test_config['tail_readout_no_act'])

        optimizer = optim.AdamW(model.parameters(),
                              lr=self.test_config['learning_rate'],
                              weight_decay=self.test_config['weight_decay'])

        # load stat dict if there exists.
        if os.path.exists(os.path.join(self.test_config['model_save_dir'],
                                       'agat_state_dict.pth')):
            try:
                checkpoint = load_state_dict(self.test_config['model_save_dir'])
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                model = model.to(self.device)
                model.device = self.device
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(
                    f'User info: Model and optimizer state dict loaded successfully from {self.test_config["model_save_dir"]}.')
            except:
                print('User warning: Exception catched when loading models and optimizer state dict.')
        else:
            print('User info: Checkpoint not detected')

        # loss function
        with torch.no_grad():
            # echo the parameter
            # for param_tensor in model.state_dict():
            #     print(param_tensor, '\t', model.state_dict()[param_tensor].size())


            contcarpath = "/mnt/e/ComPhys/GitDemo/cbfnet/enpaper/Dataset/CCData/C2H5CCCH/C2H5CCCH_1/X/0/CONTCAR"
            fig_path = os.path.join(self.test_config['output_files'], 'attention')
            if not os.path.exists(fig_path):
                os.makedirs(fig_path, exist_ok=True)

            ordinalatoms = findordinalatoms(contcarpath)
            atomconsider = ordinalatoms[0]


            database = BuildOneGraph(path_file=contcarpath, num_of_cores=1)
            graph = database.build().to(self.device)
            edge_ids = graph.edges()
            feat = graph.ndata['h']

            # h : (number of nodes, dimension of one-hot code representation)
            # graph.edata['dist'] : (number of edges)
            # dist : (number of edges, 1, 1)
            dist = torch.reshape(graph.edata['dist'], (-1, 1, 1))
            dist = torch.where(dist < 0.01, 0.01, dist)

            # shape of dist: (number of edges, number of heads, 1)
            dist = model.get_head_mechanism(model.head_list, dist)

            h, attention = model.gat_layers[0](feat, dist, graph)
            h, attention = model.gat_layers[1](h, dist, graph)
            h, attention = model.gat_layers[2](h, dist, graph)
            h, attention = model.gat_layers[3](h, dist, graph)
            h, attention = model.gat_layers[4](h, dist, graph)

            #
            # # # plot the heatmap
            # # all nodes and all edges
            # for headmapnum in range(6):
            #     adjacency_matrix = np.zeros((len(feat), len(feat)))
            #     for i in range(len(edge_ids[0])):
            #         souratom = edge_ids[0][i]
            #         destatom = edge_ids[1][i]
            #         adjacency_matrix[souratom, destatom] = attention[i, headmapnum, 0]
            #
            #     # # plot the heatmap
            #     import pandas as pd
            #     import seaborn as sns
            #     import matplotlib.pyplot as plt
            #     # mask = (adjacency_matrix == 0).astype(np.bool_)
            #     df = pd.DataFrame(adjacency_matrix)
            #     plt.figure(figsize=(15, 15))
            #     # sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, square=True, linewidths=.5)
            #     # sns.heatmap(df, cmap="viridis", fmt=".3f", mask=mask, square=True, annot=True)
            #     sns.heatmap(df, cmap="viridis", fmt=".3f", square=True, annot=True)
            #     plt.xlabel('Source atom', fontsize=15)
            #     plt.ylabel('Target atom', fontsize=15)
            #     plt.title("Attention Scores Heatmap", fontsize=30)
            #     path_fig = os.path.join(fig_path, f"headmapnum{headmapnum}.jpg")
            #     plt.savefig(path_fig, dpi=600)
            #     plt.show()
            #

            atomneighbor, columnlist = [], []
            for i, edge_id in enumerate(edge_ids[1]):
                # the message from others to edge_ids[1]
                if atomconsider == edge_id:
                    atomneighbor.append(edge_ids[0][i])
                    columnlist.append(i)
            atomneighbor = [atom.cpu().numpy() for atom in atomneighbor]
            atomneighbor = np.stack(atomneighbor)

            # calculate the entropy
            # echo the neighbor atom
            for atthead in range(6):
                attentionconsider = [attention[column, atthead, 0].cpu() for column in columnlist]
                print(f"{entropy(attentionconsider):.5f}")

            #
            # # # plot the graph with attention as edge
            # import matplotlib.pyplot as plt
            # import matplotlib.cm as cm
            # color_map = cm.plasma
            # attentionhead = attention[:, 1, 0]
            # edge_colors = color_map(((attentionhead - attentionhead.min()).cpu().numpy()) /
            #                         (attentionhead.max() - attentionhead.min()).cpu().numpy())
            # # edge_colors = color_map(attentionhead)
            #
            # nx.draw(graph.cpu().to_networkx(), with_labels=True,
            #         font_weight='bold', font_size=15, node_size=500,
            #         node_color='#67aba5',
            #         # edge_color='#7e7d73',
            #         edge_color=edge_colors,
            #         style='dotted')
            # sm = plt.cm.ScalarMappable(cmap=color_map)
            # sm.set_array([])
            # # plt.colorbar(sm, ax=plt.gca())
            # cbar = plt.colorbar(sm, ax=plt.gca(), aspect=10)
            # cbar.set_label('Attention Score')
            #
            # plt.savefig(os.path.join(fig_path, 'graph.jpg'), format='jpg', dpi=1000, bbox_inches='tight')
            # plt.show()

if __name__ == '__main__':
    score = Score(model_save_dir="../out_put/agat_model")
    score.echo()