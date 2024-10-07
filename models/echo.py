# @Time    : 2023/12/26 15:55
# @Author  : Feiyu
# @File    : Model_Test.py
# @diligent：What doesn't kill me makes me stronger.
# @Function: Base on trained model, developing automated programs to calculate the peakforce.


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
        a = self.test_config['a']
        b = self.test_config['b']
        mae = nn.L1Loss()
        r = PearsonR


    def test(self, bg):
        """test with trained model"""
        with torch.no_grad():
            energy_pred_all, force_pred_all = [], []
            energy_pred, force_pred = self.model.forward(bg)

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

                f_csv.write(evepath.strip() + ',  ' + str(trueforce) + ',  ' + str(preresultantforce) + ',  ' + str(free_energy_per_atom) + ',  ' + str(preenergy) + '\n')
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
if __name__ == '__main__':
    from data.build_dataset import BuildDatabase
    # database = BuildDatabase(path_file='../../Testset/paths.log', dataset_path="../dataset/Testset", num_of_cores=2)
    # database.build()
    t = echo(path_file='../../Dataset/paths.log', output_files='./')
    t.output()