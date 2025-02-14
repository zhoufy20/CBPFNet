# @Time    : 2023/12/26 15:55
# @Author  : Feiyu
# @File    : Model_Test.py
# @diligent：What doesn't kill me makes me stronger.
# @Function: Base on trained model, developing automated programs to calculate the peakforce.


import os
import csv
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from models.model import PotentialModel
from lib.model_lib import (PearsonR, config_parser, load_state_dict, deepmdgeneratecontcar, 
                           generatecontcar, findinistru, eneforlenoutput)
from default_parameters import  default_train_config, default_data_config

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage is deprecated.*")


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
            os.makedirs(self.test_config['output_files'])

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
        """Automate peak force prediction"""
        self.test_config = {**self.test_config, **config_parser(test_config)}
        precbpflogpath = os.path.join(self.test_config['output_files'], 'prediction.log')
        self.log = open(precbpflogpath, 'w', buffering=1)
        start_time = time.time()
        precbpforcelist, dftcbpforcelist, dpcbpforcelist = [], [], []

        """Initial DP module"""
        # dp freeze -o graph.pb
        # dp compress -i graph.pb -o graph-compress.pb
        # dp test -m graph-compress.pb -s ../traindp -n 40 -d trainresults
        if self.test_config['deepmd']:
            from deepmd.calculator import DP
            calc = DP(model="deepmd/deepmd_gpu/graph-compress.pb")

        """find every /0/ path"""
        inipaths = findinistru(self.test_config["path_file"])
        current_directory = os.getcwd()
        f_csv_path = os.path.join(self.test_config['output_files'], 'automate.csv')
        f = open(f_csv_path, 'w', newline='')
        f_csv = csv.writer(f)
        if self.test_config['deepmd']:    
            if os.path.getsize(f_csv_path) == 0:
                f_csv.writerow(['Index', 'Deepmd-Predict-peak-force', 'Predict-peak-force', 'DFT-peak-force', 'Deepmd-Relative-error', 'Relative-error', 'Inipath'])
        else:
            if os.path.getsize(f_csv_path) == 0:
                f_csv.writerow(['Index', 'Predict-peak-force', 'DFT-peak-force', 'Relative-error', 'Inipath'])
        print('========================================================================', file=self.log)
        print(self.model, file=self.log)
        print('========================================================================', file=self.log)
        print("Epoch    dppeakforce   prepeakforce  truepeakforce   error   realativeerror  Dur_(s) Train_info", file=self.log)
        
        # inipath:Testset/C2H5CCCH_2/0/
        for index, inipath in enumerate(inipaths):
            # prepare out file
            outpath = os.path.abspath(os.path.join(self.test_config['output_files'], 'poscarall', f'poscar{index}'))
            preresultantforcelist = []
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            # find the initial contcar to automate
            contcarpath = os.path.join(inipath, "CONTCAR")
            with open(contcarpath, 'r') as file:

                if self.test_config['deepmd']:
                    bglist, ordinalatoms, strainlist, dpforcelist = deepmdgeneratecontcar(os.path.abspath(outpath), contcarpath, calc)

                bglist, ordinalatoms, strainlist = generatecontcar(os.path.abspath(outpath), contcarpath)    
                for i, bg in enumerate(bglist):
                    bg = bg.to(self.device)
                    force_pred_all, energy_pred_all = self.test(bg)
                    forceatomstre = force_pred_all[ordinalatoms[0]]
                    resultantforce = torch.norm(forceatomstre)
                    preresultantforcelist.append(resultantforce.cpu().numpy())
            # CBPForce prediction
            prepeakforce = max(preresultantforcelist)
            precbpforcelist.append(prepeakforce)
            # DFT CBPForce 
            eneforlenarray, truestrainlist = eneforlenoutput(os.path.abspath(os.path.join(inipath, "..")))
            truepeakforce = max(eneforlenarray[1,:])
            dftcbpforcelist.append(truepeakforce)
            # DP CBPForce
            if self.test_config['deepmd']:
                dppeakforce = max(dpforcelist)
                dpcbpforcelist.append(dppeakforce)

            outputtruepath = os.path.join(outpath, 'dftforce.csv')
            outputprepath = os.path.join(outpath, 'preforce.csv')
            if self.test_config['deepmd']:
                outputpredppath = os.path.join(outpath, 'dpforce.csv')

            with open(outputtruepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for true_strain, trforce in zip(truestrainlist, eneforlenarray[1, :]):
                    writer.writerow([true_strain, trforce])
            with open(outputprepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for strain, pre_force in zip(strainlist, preresultantforcelist):
                    writer.writerow([strain, pre_force])
            if self.test_config['deepmd']:        
                with open(outputpredppath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for strain, dp_force in zip(strainlist, dpforcelist):
                        writer.writerow([strain, dp_force])


            error = prepeakforce-truepeakforce
            if self.test_config['deepmd']:
                relaerrdp_dft = (dppeakforce-truepeakforce)/truepeakforce
            relaerrnet_dft = (prepeakforce-truepeakforce)/truepeakforce
            dur = time.time() - start_time
            if self.test_config['deepmd']:
                print("{:0>5d} {:1.8f} {:1.8f} {:1.8f} {:1.8f} {:1.8f} {:10.1f} Prediction_info".format(
                    index, dppeakforce, prepeakforce, truepeakforce, relaerrdp_dft, relaerrnet_dft, dur), file=self.log)
                print("{:0>5d} {:1.8f} {:1.8f} {:1.8f} {:1.8f} {:1.8f} {:10.1f} Prediction_info".format(
                    index, dppeakforce, prepeakforce, truepeakforce, relaerrdp_dft, relaerrnet_dft, dur))
            else:
                print("{:0>5d} {:1.8f} {:1.8f} {:1.8f} {:10.1f} Prediction_info".format(
                    index, prepeakforce, truepeakforce, relaerrnet_dft, dur), file=self.log)
                print("{:0>5d} {:1.8f} {:1.8f} {:1.8f} {:10.1f} Prediction_info".format(
                    index, prepeakforce, truepeakforce, relaerrnet_dft, dur))

            if self.test_config['deepmd']:
                data_row = [f'POSCAR{index}', dppeakforce, prepeakforce, truepeakforce, relaerrdp_dft, relaerrnet_dft, inipath]
            else:
                data_row = [f'POSCAR{index}', prepeakforce, truepeakforce, relaerrnet_dft, inipath]
            
            f_csv.writerow(data_row)                    

            # plot the force-strain about truepeakforce and predictionpeakforce
            plt.figure(figsize=(10, 6))
            plt.plot(strainlist, preresultantforcelist, label='Prediction', color='blue', marker='o', linestyle='-', linewidth=2, markersize=8)
            if self.test_config['deepmd']:
                plt.plot(strainlist, dpforcelist, label='DeepMD-kit', color='black', marker='s', linestyle='--', linewidth=2, markersize=8)
            plt.plot(truestrainlist, eneforlenarray[1, :], label='Truth', color='green', marker='s', linestyle='--', linewidth=2, markersize=8)
            plt.xlabel('Stretch strain', fontsize=14)
            plt.ylabel('Force of prediction and true', fontsize=14)
            plt.title(f"POSCAR{index}-Force-Strain Plot", fontsize=16)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend()
            plt.savefig(f"forces{index}.png", dpi=300)
            plt.close()

            os.chdir(current_directory)

        """print error between precbpforcelist and dftcbpforcelist"""
        mae = nn.L1Loss()
        r = PearsonR
        precbpforcelist = [arr.item() for arr in precbpforcelist]
        dftcbpforcelist = [arr.item() for arr in dftcbpforcelist]
        if self.test_config['deepmd']:
            dpcbpforcelist = [arr.item() for arr in dpcbpforcelist]
        preforce_tensor = torch.tensor(precbpforcelist)
        trueforce_tensor = torch.tensor(dftcbpforcelist)
        if self.test_config['deepmd']:
            dpforce_tensor = torch.tensor(dpcbpforcelist)
        pre_force_mae = mae(preforce_tensor, trueforce_tensor)
        pre_force_r = r(preforce_tensor, trueforce_tensor)
        if self.test_config['deepmd']:    
            dp_force_mae = mae(dpforce_tensor, trueforce_tensor)
            dp_force_r = r(dpforce_tensor, trueforce_tensor)        
        print(f'''CBPFNet_Force_MAE : {pre_force_mae.item()}  CBPFNet_Force_R : {pre_force_r.item()}''')
        if self.test_config['deepmd']:
            print(f'''Deepmd_Force_MAE : {dp_force_mae.item()}   Deepmd_Force_R : {dp_force_r.item()}''')
        
if __name__ == '__main__':
    from data.build_dataset import BuildDatabase
    # database = BuildDatabase(path_file='../../Testset/paths.log', dataset_path="../dataset/Testset", num_of_cores=2)
    # database.build()
    t = Test(path_file='../../Dataset/paths.log', dataset_path='../dataset/Testset/all_graphs.bin', output_files='../out_put/predict')
    t.output()