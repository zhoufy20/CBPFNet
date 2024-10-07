# @Time    : 2023/12/26 15:55
# @Author  : Feiyu
# @File    : Model_Test.py
# @diligent：What doesn't kill me makes me stronger.
# @Function: Base on trained model, developing automated programs to calculate the peakforce.


import os
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from models.model import PotentialModel
from lib.model_lib import (PearsonR, config_parser, load_state_dict, generatecontcar, findinistru, eneforlenoutput)
from default_parameters import  default_train_config, default_data_config


class Test(object):
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
        """Automate peak force prediction"""
        self.test_config = {**self.test_config, **config_parser(test_config)}
        predictionlogpath = os.path.join(self.test_config['output_files'], 'prediction.log')
        self.log = open(predictionlogpath, 'w', buffering=1)
        start_time = time.time()

        inipaths = findinistru(self.test_config["path_file"])
        current_directory = os.getcwd()
        f_csv = open(os.path.join(self.test_config['output_files'], 'fname_path.csv'), 'w', buffering=1)
        print('========================================================================', file=self.log)
        print(self.model, file=self.log)
        print('========================================================================', file=self.log)
        print("Epoch  prepeakforce  truepeakforce  error  realativeerror  Dur_(s)  Train_info", file=self.log)
        # inipath:Testset/C2H5CCCH_2/0/
        for index, inipath in enumerate(inipaths):
            preresultantforcelist = []
            # prepare out file
            outpath = os.path.abspath(os.path.join(self.test_config['output_files'], 'poscarall', f'poscar{index}'))
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            contcarpath = os.path.join(inipath, "CONTCAR")
            with open(contcarpath, 'r') as file:
                bglist, ordinalatoms, strainlist = generatecontcar(os.path.abspath(outpath), contcarpath)
                for i, bg in enumerate(bglist):
                    bg = bg.to(self.device)
                    force_pred_all, energy_pred_all = self.test(bg)
                    forceatomstre = force_pred_all[ordinalatoms[1]]
                    resultantforce = torch.norm(forceatomstre)
                    preresultantforcelist.append(resultantforce.cpu().numpy())
            prepeakforce = max(preresultantforcelist)
            eneforlenarray, truestrainlist = eneforlenoutput(os.path.abspath(os.path.join(inipath, "..")))
            truepeakforce = max(eneforlenarray[1,:])
            error = prepeakforce-truepeakforce
            realativeerror = error/truepeakforce
            dur = time.time() - start_time
            print("{:0>5d} {:1.8f} {:1.8f} {:1.8f} {:1.8f} {:10.1f} Train_info".format(
                index, prepeakforce, truepeakforce, error, realativeerror, dur), file=self.log)
            print("{:0>5d} {:1.8f} {:1.8f} {:1.8f} {:1.8f} {:10.1f} Train_info".format(
                index, prepeakforce, truepeakforce, error, realativeerror, dur))

            f_csv.write(f'POSCAR{index}' + ',  ' + inipath + ',  ')
            f_csv.write(str(prepeakforce) + ',  ' + str(truepeakforce) + ',  ' + str(error) +',  ' + str(realativeerror) +'\n')

            # plot the force-strain about truepeakforce and predictionpeakforce
            plt.figure(figsize=(10, 6))
            plt.plot(strainlist, preresultantforcelist, label='Prediction', color='blue', marker='o', linestyle='-', linewidth=2, markersize=8)
            plt.plot(truestrainlist, eneforlenarray[1, :], label='Truth', color='green', marker='s', linestyle='--', linewidth=2, markersize=8)
            plt.xlabel('Stretch strain', fontsize=14)
            plt.ylabel('Force of prediction and true', fontsize=14)
            plt.title(f"POSCAR{index}-Force-Strain Plot", fontsize=16)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend()
            plt.savefig(f"forces{index}.png", dpi=300)
            # plt.show()
            plt.close()

            os.chdir(current_directory)

if __name__ == '__main__':
    from data.build_dataset import BuildDatabase
    # database = BuildDatabase(path_file='../../Testset/paths.log', dataset_path="../dataset/Testset", num_of_cores=2)
    # database.build()
    t = Test(path_file='../../Dataset/paths.log', dataset_path='../dataset/Testset/all_graphs.bin', output_files='../out_put/predict')
    t.output()