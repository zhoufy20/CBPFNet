# @Time    : 2023/12/26 15:55
# @Author  : Feiyu
# @File    : Model_Test.py
# @diligent：What doesn't kill me makes me stronger.
# @Function: echo every prediction and true force&energy.


import os
import time
import csv
import ase
from ase.io import read
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from models.model import PotentialModel
from lib.model_lib import (PearsonR, config_parser, load_state_dict, generatecontcar, findinistru, eneforlenoutput, findordinalatoms)
from default_parameters import  default_train_config, default_data_config

class echo(object):
    """automated programs to calculate the peakforce"""
    def __init__(self, **test_config):
        self.test_config = {**default_data_config, **default_train_config, **config_parser(test_config)}

        # check device
        self.device = self.test_config['device']
        if torch.cuda.is_available() and self.device == 'cpu':
            print('User warning: `CUDA` device is available, but you choose `cpu`.')
        elif not torch.cuda.is_available() and self.device.split(':')[0] == 'cuda':
            print('User warning: `CUDA` device is not available, but you choose `cuda:0`. Change the device to `cpu`.')
            self.device = 'cpu'
        print('User info: Specified device for potential models:', self.device)

        # read dataset
        print(f'User info: Loading dataset from {self.test_config["path_file"]}')

        if not os.path.exists(self.test_config['output_files']):
            os.makedirs(self.test_config['output_files'], exist_ok=True)

        # load the saved model parameters
        self.model = PotentialModel(self.test_config['gat_node_dim_list'],
                               self.test_config['energy_readout_node_list'],
                               self.test_config['force_readout_node_list'],
                               self.test_config['head_list'],
                               self.test_config['bias'],
                               self.test_config['negative_slope'],
                               self.device,
                               self.test_config['tail_readout_no_act'])
        optimizer = optim.AdamW(self.model.parameters(),
                              lr=self.test_config['learning_rate'],
                              weight_decay=self.test_config['weight_decay'])

        # load stat dict if there exists.
        if os.path.exists(os.path.join(self.test_config['model_save_dir'], 'agat_state_dict.pth')):
            try:
                checkpoint = load_state_dict(self.test_config['model_save_dir'])
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                model = self.model.to(self.device)
                model.device = self.device
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(
                    f'User info: Model and optimizer state dict loaded successfully from {self.test_config["model_save_dir"]}.')
            except:
                print('User warning: Exception catched when loading models and optimizer state dict.')
        else:
            print('User info: Checkpoint not detected')

        # loss function
        criterion = self.test_config['criterion']
        a, b, c = self.train_config['a'], self.train_config['b'], self.train_config['c']

        mae = nn.L1Loss()
        r = PearsonR


    def test(self, bg):
        """test with trained model"""
        with torch.no_grad():
            energy_pred_all, force_pred_all = [], []
            energy_pred, force_pred, _ = self.model.forward(bg)

            energy_pred_all.append(energy_pred)
            force_pred_all.append(force_pred)

            energy_pred_all = torch.cat(energy_pred_all)
            force_pred_all = torch.cat(force_pred_all)

        return force_pred_all, energy_pred_all

    def output(self, **test_config):
        self.test_config = {**self.test_config, **config_parser(test_config)}
        f_csv = open(os.path.join(self.test_config['output_files'], 'echo.csv'), 'w', buffering=1)
        trueforcelist, preforcelist=[], []
        truenergylist, preenergylist=[], []
        mae = nn.L1Loss()
        r = PearsonR

        current_directory = os.getcwd()
        print('========================================================================')
        print(self.model)
        print('========================================================================')
        print("Num  trueforce  preforce  trueene  preene  Dur_(s)  Train_info")
        # inipath:Testset/C2H5CCCH_2/0/
        with open(self.test_config["path_file"], 'r') as file:
            for index, evepath in enumerate(file):
                contcarpath = os.path.join(evepath.strip(), "CONTCAR")
                outcarpath = os.path.join(evepath.strip(), "OUTCAR")
                frames_contcar = read(contcarpath, index='-1:')
                atoms = frames_contcar[-1]
                frames_outcar = read(outcarpath, index='-1:')

                # calc the original bond length
                ordinalatoms = findordinalatoms(contcarpath)

                from data.build_dataset import BuildOneGraph
                database = BuildOneGraph(path_file=contcarpath, num_of_cores=1)
                bg = database.build()

                bg = bg.to(self.device)
                force_pred_all, energy_pred_all = self.test(bg)
                forceatomstre = force_pred_all[ordinalatoms[0]]
                preresultantforce = torch.norm(forceatomstre).item()
                preenergy = (energy_pred_all.item())
                preforcelist.append(preresultantforce)
                preenergylist.append(preenergy)

                num_atoms = len(frames_outcar[0])
                free_energy = [x.get_total_energy() for x in frames_outcar]
                free_energy_per_atom = free_energy[0] / num_atoms
                truenergylist.append(free_energy_per_atom)

                trueforces = frames_outcar[0].get_forces(apply_constraint=False)
                trueforce = np.linalg.norm(trueforces[ordinalatoms[0], :])
                trueforcelist.append(trueforce)

                f_csv.write(evepath.strip() + ',  ' + str(trueforce) + ',  ' + str(preresultantforce) + ',  '
                            + str(free_energy_per_atom) + ',  ' + str(preenergy) + '\n')
                print("{:0>5d} {:1.8f} {:1.8f} {:1.8f} {:1.8f}".format(
                index, trueforce, preresultantforce, free_energy_per_atom, preenergy))
            os.chdir(current_directory)
        trueforcelist = torch.tensor([trueforcelist])  # Replace with actual true force values
        preforcelist = torch.tensor([preforcelist])  # Replace with actual predicted force values
        truenergylist = torch.tensor([truenergylist])
        preenergylist = torch.tensor([preenergylist])
        force_mae = mae(trueforcelist, preforcelist)
        force_r = r(trueforcelist, preforcelist)
        energy_mae=mae(truenergylist, preenergylist)
        energy_r=r(truenergylist, preenergylist)
        print(f'''Force_MAE  : {force_mae.item()},  Force_R    : {force_r.item()}''')
        print(f'''Energy_MAE  : {energy_mae.item()},  Energy_R    : {energy_r.item()}''')

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
    from data.build_dataset import BuildDatabase
    # database = BuildDatabase(path_file='../../Testset/paths.log', dataset_path="../dataset/Testset", num_of_cores=2)
    # database.build()
    t = echo(path_file='../../Dataset/paths.log', output_files='./')
    t.output()