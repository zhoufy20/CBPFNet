# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 23:07:24 2023

@author: ZHANG Jun
"""

""" Configurations for database construction, training process, and prediction behaviors. """

import os
import torch.nn as nn

default_elements = ['Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'B',  'Ba',
                    'Be', 'Bh', 'Bi', 'Bk', 'Br', 'C',  'Ca', 'Cd', 'Ce', 'Cf',
                    'Cl', 'Cm', 'Cn', 'Co', 'Cr', 'Cs', 'Cu', 'Db', 'Ds', 'Dy',
                    'Er', 'Es', 'Eu', 'F',  'Fe', 'Fl', 'Fm', 'Fr', 'Ga', 'Gd',
                    'Ge', 'H',  'He', 'Hf', 'Hg', 'Ho', 'Hs', 'I',  'In', 'Ir',
                    'K',  'Kr', 'La', 'Li', 'Lr', 'Lu', 'Lv', 'Mc', 'Md', 'Mg',
                    'Mn', 'Mo', 'Mt', 'N',  'Na', 'Nb', 'Nd', 'Ne', 'Nh', 'Ni',
                    'No', 'Np', 'O',  'Og', 'Os', 'P',  'Pa', 'Pb', 'Pd', 'Pm',
                    'Po', 'Pr', 'Pt', 'Pu', 'Ra', 'Rb', 'Re', 'Rf', 'Rg', 'Rh',
                    'Rn', 'Ru', 'S',  'Sb', 'Sc', 'Se', 'Sg', 'Si', 'Sm', 'Sn',
                    'Sr', 'Ta', 'Tb', 'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'Ts',
                    'U',  'V',  'W',  'Xe', 'Y',  'Yb', 'Zn', 'Zr']

default_build_properties = {'energy': True,
                         'forces': True,
                         'cell': False,
                         'cart_coords': False,
                         'frac_coords': False,
                         'constraints': False,
                         'stress': False,
                         'distance': True,
                         'direction': True,
                         'path': False}

default_data_config =  {
    'species': default_elements,
    'path_file': 'paths.log', # A file of absolute paths where OUTCAR and XDATCAR files exist.
    'build_properties': default_build_properties, # Properties needed to be built into graph.
    'topology_only': False,
    'dataset_path': 'dataset', # Path where the collected data to save.
    'mode_of_NN': 'ase_dist', # How to identify connections between atoms. 'ase_natural_cutoffs', 'pymatgen_dist', 'ase_dist', 'voronoi'. Note that pymatgen is much faster than ase.
    'cutoff': 2.0, # Cutoff distance to identify connections between atoms. Deprecated if ``mode_of_NN`` is ``'ase_natural_cutoffs'``
    'load_from_binary': False, # Read graphs from binary graphs that are constructed before. If this variable is ``True``, these above variables will be depressed.
    'num_of_cores': 8,
    'super_cell': False,
    'keep_readable_structural_files': False,
    'mask_similar_frames': True,
    'mask_reversed_magnetic_moments': False, # or -0.5 # Frames with atomic magnetic moments lower than this value will be masked.
    'energy_stride': 0.005,
    'scale_prop': False
             }

FIX_VALUE = [1,3]

default_train_config = {
    'verbose': 1, # `0`: no train and validation output; `1`: Validation and test output; `2`: train, validation, and test output.
    'dataset_path': os.path.join('dataset', 'Dataset', 'all_graphs.bin'),
    'model_save_dir': os.path.join('out_put', 'agat_model'),
    'epochs': 8000,
    'output_files': os.path.join('out_put', 'train'),
    'device': 'cuda:0',
    # 'device': 'cpu',

    'validation_size': 0.20,
    'test_size': 0.02,
    'early_stop': True,
    'stop_patience': 1000,
    'head_list': ['mul', 'div', 'free'],
    'gat_node_dim_list': [len(default_elements), 128, 128, 164, 164, 200, 200, 200],
    'energy_readout_node_list': [600, 600, 500, 300, 150, 100, 50, 25, 10, FIX_VALUE[0]],
    'force_readout_node_list': [600, 600, 500, 300, 150, 100, 50, 25, 10, FIX_VALUE[1]],
    'bias': True,
    'negative_slope': 0.2,
    'criterion': nn.MSELoss(),
    'a': 1.0,
    'b': 10000.0,
    'optimizer': 'adam',
    'learning_rate': 0.0001,
    'weight_decay': 0.0, # weight decay (L2 penalty)
    'batch_size': 256,
    'val_batch_size': 256,
    'transfer_learning': False,
    'trainable_layers': -3,
    'mask_fixed': False,
    'tail_readout_no_act': [3, 3],}



