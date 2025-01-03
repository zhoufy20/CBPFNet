#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :atomic_feature.py
# @Time      :2024/3/24 13:51
# @Author    :Feiyu
# @Main      :Extra Vasp Files and construct a dgl graph using ase package
#===================================================================================================
#### reference from DOI:https://doi.org/10.1016/j.joule.2023.06.003
#### reference code from url:https://github.com/jzhang-github/AGAT
###  cite:Zhang, Jun & Wang, Chaohui & Huang, Shasha & Xiang, Xuepeng & Xiong, Yaoxu & Biao, Xu
###  & Ma, Shihua & Fu, Haijun & Kai, Jijung & Kang, Xiongwu & Zhao, Shijun. (2023).
###  Design high-entropy electrocatalyst via interpretable deep graph attention learning.
###  Joule. 7. 10.1016/j.joule.2023.06.003.
#===================================================================================================

import numpy as np
import sys
import os
import json
import multiprocessing

from ase.io import read, write
import ase
from ase.neighborlist import natural_cutoffs
import torch
import dgl
from dgl.data.utils import save_graphs, load_graphs
from tqdm import tqdm

from default_parameters import default_data_config
from data.atomic_feature import get_atomic_feature_onehot, get_atomic_features
from lib.model_lib import config_parser

class CrystalGraph(object):
    """ Read the crystal graph and return a ``DGL.graph``."""
    """
    . Hint::
        Although we recommend representing atoms with one hot code, you can
        use the another way with: ``self.all_atom_feat = get_atomic_features()``

    . Hint::
        In order to build a reasonable graph, a samll cell should be repeated.
        One can modify "self._cell_length_cutoff" for special needs.
        
    .. Hint::
        We encourage you to use ``ase`` module to build crystal graphs.
        The ``pymatgen`` module needs some dependencies that conflict with
        other modules.
    """
    def __init__(self, **data_config):
        self.data_config = {**default_data_config, **config_parser(data_config)}

        # check inputs
        if self.data_config['topology_only']:
            self.data_config['build_properties']['energy'] = False
            self.data_config['build_properties']['forces'] = False
            self.data_config['build_properties']['cell'] = False
            self.data_config['build_properties']['stress'] = False
            self.data_config['build_properties']['frac_coords'] = False

        """ Although we recommend representing atoms with one hot code, you can
        use the another way with the next line: """
        self.all_atom_feat = get_atomic_features(self.data_config['species'])
        # self.all_atom_feat = get_atomic_feature_onehot(self.data_config['species'])


        """ In order to build a reasonable graph, a samll cell should be repeated.
        One can modify "self._cell_length_cutoff" for special needs. """
        self._cell_length_cutoff = np.array([11.0, 5.5, 0.0]) # Unit: angstrom. # increase cutoff to repeat more.

        if self.data_config['mode_of_NN'] == 'voronoi' or self.data_config['mode_of_NN'] == 'pymatgen_dist':
            ''' We encourage you to use ``ase`` module to build crystal graphs.
            The ``pymatgen`` module may need dependencies that conflict with
            other modules.'''
            try:
                from pymatgen.core.structure import Structure
                from agat.data.PymatgenStructureAnalyzer import VoronoiConnectivity
            except ModuleNotFoundError:
                raise ModuleNotFoundError('We encourage you to use ``ase`` module \
                to build crystal graphs. The ``pymatgen`` module needs some \
                dependencies that conflict with other modules. Install\
                ``pymatgen`` with ``pip install pymatgen``.')

            self.Structure = Structure
            self.VoronoiConnectivity = VoronoiConnectivity

        # self.device = self.data_config['device']

        if self.data_config['build_properties']['path']:
            print('You choose to store the path of each graph. '
                  'Be noted that this will expose your file structure when you publish your dataset.')
        self.dtype = torch.float


    def get_crystal(self, crystal_fpath):
        """ Read structural file and return a pymatgen crystal object. """
        assert os.path.exists(crystal_fpath), str(crystal_fpath) + " not found."
        structure = self.Structure.from_file(crystal_fpath)
        """If there is only one site in the cell, this site has no other
        neighbors. Thus, the graph cannot be built without repetition."""
        cell_span = structure.lattice.matrix.diagonal()
        if cell_span.min() < max(self._cell_length_cutoff) and self.data_config['super_cell']:
            repeat = np.digitize(cell_span, self._cell_length_cutoff) + 1
            structure.make_supercell(repeat)
        return structure # pymatgen format

    def get_1NN_pairs_voronoi(self, crystal):
        """
        .. py:method:: get_1NN_pairs_voronoi(self, crystal)

           The ``get_connections_new()`` of ``VoronoiConnectivity`` object is modified.

           :param pymatgen.core.structure crystal: a pymatgen structure object.
           :Returns:
              - index of senders
              - index of receivers
              - a list of distance between senders and receiver
        """

        crystal_connect = self.VoronoiConnectivity(crystal, self.data_config['cutoff'])
        return crystal_connect.get_connections_new() # the outputs are bi-directional and self looped.

    def get_1NN_pairs_distance(self, crystal):
        """
        .. py:method:: get_1NN_pairs_distance(self, crystal)

           Find the index of senders, receivers, and distance between them based on the ``distance_matrix`` of pymargen crystal object.

           :param pymargen.core.structure crystal: pymargen crystal object
           :Returns:
              - index of senders
              - index of receivers
              - a list of distance between senders and receivers
        """

        distance_matrix = crystal.distance_matrix
        sender, receiver = np.where(distance_matrix < self.data_config['cutoff'])
        dist = distance_matrix[(sender, receiver)]
        return sender, receiver, dist # the outputs are bi-directional and self looped.

    def get_1NN_pairs_ase_distance(self, ase_atoms):
        """
        .. py:method:: get_1NN_pairs_ase_distance(self, ase_atoms)

           :param ase.atoms ase_atoms: ``ase.atoms`` object.
           :Returns:
              - index of senders
              - index of receivers
              - a list of distance between senders and receivers
        """

        distance_matrix = ase_atoms.get_all_distances(mic=True)
        sender, receiver = np.where(distance_matrix < self.data_config['cutoff'])
        dist = distance_matrix[(sender, receiver)]
        return sender, receiver, dist # the outputs are bi-directional and self looped.

    def get_ndata(self, crystal):
        """ the atomic representations of a crystal graph. """
        ndata = []
        if isinstance(crystal, ase.Atoms):
            # num_sites = len(crystal)
            for i in crystal.get_chemical_symbols():
                ndata.append(self.all_atom_feat[i])
        else: # isinstance(crystal, self.Structure)
            num_sites = crystal.num_sites
            for i in range(num_sites):
                ndata.append(self.all_atom_feat[crystal[i].species_string])
        return np.array(ndata)

    def get_graph_from_ase(self, fname): # ase_atoms or file name
        """ return a bidirectional graph with self-loop connection and a dict of information of graph-level features. """
        assert not isinstance(fname, ase.Atoms) or not self.data_config['build_properties']['forces'], \
            'The input cannot be a ase.Atoms object when include_forces is ``True``, try with an ``OUTCAR``'

        if not isinstance(fname, ase.Atoms):
            ase_atoms = read(fname)
        else:
            ase_atoms = fname
        num_sites = len(ase_atoms)

        if self.data_config['mode_of_NN'] == 'ase_dist':
            assert self.data_config['cutoff'], ('The `cutoff` cannot be `None` '
                                                'in this case. Provide a float here.')
            ase_cutoffs = self.data_config['cutoff']
        elif self.data_config['mode_of_NN'] == 'ase_natural_cutoffs':
            ase_cutoffs = np.array(natural_cutoffs(ase_atoms))

        i, j, d, D = ase.neighborlist.neighbor_list('ijdD', # i: sender; j: receiver; d: distance; D: direction
                                                    ase_atoms,
                                                    cutoff=ase_cutoffs,
                                                    self_interaction=True)
        ndata                   = self.get_ndata(ase_atoms)
        bg                      = dgl.graph((i, j))
        bg.ndata['h']           = torch.tensor(ndata, dtype=self.dtype)
        if self.data_config['build_properties']['distance']:
            bg.edata['dist']        = torch.tensor(d, dtype=self.dtype)
        if self.data_config['build_properties']['direction']:
            bg.edata['direction']   = torch.tensor(D, dtype=self.dtype)
        if self.data_config['build_properties']['constraints']:
            constraints             = [[1, 1, 1]] * num_sites
            for c in ase_atoms.constraints:
                if isinstance(c, ase.constraints.FixScaled):
                    constraints[c.a] = c.mask

                elif isinstance(c, ase.constraints.FixAtoms):
                    for i in c.index:
                        constraints[i] = [0, 0, 0]
                elif isinstance(c, ase.constraints.FixBondLengths):
                    pass
                else:
                    raise TypeError(f'Wraning!!! Undefined constraint type: {type(c)}')
            bg.ndata['constraints'] = torch.tensor(constraints, dtype=self.dtype)
        if self.data_config['build_properties']['forces']:
            forces_true             = torch.tensor(np.load(fname+'_force.npy'), dtype=self.dtype)
            bg.ndata['forces_true'] = forces_true
        if self.data_config['build_properties']['cart_coords']:
            bg.ndata['cart_coords'] = torch.tensor(ase_atoms.positions, dtype=self.dtype)
        if self.data_config['build_properties']['frac_coords']:
            bg.ndata['frac_coords'] = torch.tensor(ase_atoms.get_scaled_positions(), dtype=self.dtype)

        graph_info = {}
        if self.data_config['build_properties']['energy']:
            energy_true = torch.tensor(np.load(fname+'_energy.npy'), dtype=self.dtype)
            graph_info['energy_true'] = energy_true
        if self.data_config['build_properties']['stress']:
            stress_true = torch.tensor(np.load(fname+'_stress.npy'), dtype=self.dtype)
            graph_info['stress_true'] = stress_true
        if self.data_config['build_properties']['cell']:
            cell_true = torch.tensor(ase_atoms.cell.array, dtype=self.dtype)
            graph_info['cell_true'] = cell_true
        if self.data_config['build_properties']['path']:
            graph_info['path'] = fname
        return bg, graph_info

    def get_graph_from_pymatgen(self, crystal_fname):
        """
        .. py:method:: get_graph_from_pymatgen(self, crystal_fname)

           Build graphs with pymatgen.

           :param str crystal_fname: File name.
           :return: A bidirectional graph with self-loop connection.
           :return: A dict of information of graph-level features.
        """
        assert bool(not bool(self.data_config['super_cell'] and self.data_config['build_properties']['forces'])), 'include_forces cannot be True when super_cell is True.'
        # https://docs.dgl.ai/guide_cn/graph-feature.html 
        mycrystal               = self.get_crystal(crystal_fname)
        # self.cart_coords       = mycrystal.cart_coords           # reserved for other functions

        if self.data_config['mode_of_NN'] == 'voronoi':
            sender, receiver, dist  = self.get_1NN_pairs_voronoi(mycrystal)   # the dist array is bidirectional and self-looped. # dist需要手动排除cutoff外的邻居
            location                = np.where(np.array(dist) > self.data_config['cutoff'])
            sender                  = np.delete(np.array(sender), location)
            receiver                = np.delete(np.array(receiver), location)
            dist                    = np.delete(np.array(dist), location)
        elif self.data_config['mode_of_NN'] == 'pymatgen_dist':
            sender, receiver, dist  = self.get_1NN_pairs_distance(mycrystal)

        ndata                   = self.get_ndata(mycrystal)
        bg                      = dgl.graph((sender, receiver))
        bg.ndata['h']           = torch.tensor(ndata, dtype=self.dtype)

        if self.data_config['build_properties']['distance']:
            bg.edata['dist']        = torch.tensor(dist, dtype=self.dtype)
        if self.data_config['build_properties']['cart_coords']:
            bg.ndata['cart_coords'] = torch.tensor(mycrystal.cart_coords, dtype=self.dtype)
        if self.data_config['build_properties']['frac_coords']:
            bg.ndata['frac_coords'] = torch.tensor(mycrystal.frac_coords, dtype=self.dtype)
        if self.data_config['build_properties']['direction']:
            frac_coords             = mycrystal.frac_coords
            sender_frac_coords      = frac_coords[sender]
            receiver_frac_coords    = frac_coords[receiver]
            delta_frac_coords       = receiver_frac_coords - sender_frac_coords
            image_top               = np.where(delta_frac_coords < -0.5, -1.0, 0.0)
            image_bottom            = np.where(delta_frac_coords >  0.5,  1.0, 0.0)
            image                   = image_top + image_bottom
            sender_frac_coords     += image
            sender_cart_coords      = mycrystal.lattice.get_cartesian_coords(sender_frac_coords)
            receiver_cart_coords    = mycrystal.cart_coords[receiver]
            real_direction          = receiver_cart_coords - sender_cart_coords
            direction_norm          = np.linalg.norm(real_direction, axis=1, keepdims=True)
            direction_normalized    = real_direction / direction_norm
            direction_normalized[np.where(sender == receiver)] = 0.0
            bg.edata['direction']   = torch.tensor(direction_normalized, dtype=self.dtype)
        if self.data_config['build_properties']['constraints']:
            try:
                bg.ndata['constraints'] = torch.tensor([x.selective_dynamics for x in mycrystal], dtype=self.dtype)
            except AttributeError:
                bg.ndata['constraints'] = torch.tensor([[True, True, True] for x in mycrystal], dtype=self.dtype)
        if self.data_config['build_properties']['forces']:
            forces_true             = torch.tensor(np.load(crystal_fname+'_force.npy'), dtype=self.dtype)
            bg.ndata['forces_true'] = torch.tensor((forces_true), dtype=self.dtype)

        graph_info = {}
        if self.data_config['build_properties']['energy']:
            energy_true = torch.tensor(np.load(crystal_fname+'_energy.npy'), dtype=self.dtype)
            graph_info['energy_true'] = torch.tensor((energy_true), dtype=self.dtype)
        if self.data_config['build_properties']['stress']:
            stress_true = torch.tensor(np.load(crystal_fname+'_stress.npy'), dtype=self.dtype)
            graph_info['stress_true'] = torch.tensor((stress_true), dtype=self.dtype)
        if self.data_config['build_properties']['cell']:
            cell_true = torch.tensor(mycrystal.lattice.matrix, dtype=self.dtype)
            graph_info['cell_true'] = torch.tensor((cell_true), dtype=self.dtype)
        if self.data_config['build_properties']['path']:
            graph_info['path'] = crystal_fname
        return bg, graph_info

    def get_graph(self, crystal_fname):
        """ choose which graph-construction method is used, and return a bidirectional graph with self-loop connection. """
        if self.data_config['mode_of_NN'] == 'voronoi' or self.data_config['mode_of_NN'] == 'pymatgen_dist':
            return self.get_graph_from_pymatgen(crystal_fname)
        elif self.data_config['mode_of_NN'] == 'ase_natural_cutoffs' or self.data_config['mode_of_NN'] == 'ase_dist':
            return self.get_graph_from_ase(crystal_fname)

class ReadGraphs(object):
    """ Read all graphs specified in the fname_prop.csv and saved in ['dataset_path'] all_graphs.bin """
    def __init__(self, **data_config):
        self.data_config = {**default_data_config, **config_parser(data_config)}
        # 'dataset_path': 'dataset', # Path where the collected data to save.
        assert os.path.exists(self.data_config['dataset_path']), str(self.data_config['dataset_path']) + " not found."

        self.cg               = CrystalGraph(**self.data_config)
        fname_prop_data       = np.loadtxt(os.path.join(self.data_config['dataset_path'], 'fname_prop.csv'), dtype=str, delimiter=',')
        self.name_list        = fname_prop_data[:,0]
        self.number_of_graphs = len(self.name_list)

    def read_batch_graphs(self, batch_index_list, batch_num):
        batch_fname  = [self.name_list[x] for x in batch_index_list]
        batch_g, batch_graph_info = [], []
        for fname in tqdm(batch_fname, desc='Reading ' + str(batch_num) + ' batch graphs', delay=batch_num):
            g, graph_info = self.cg.get_graph(os.path.join(self.data_config['dataset_path'] ,fname))
            batch_g.append(g)
            batch_graph_info.append(graph_info)

        batch_labels = {}
        for key in batch_graph_info[0]:
            if key != 'path':
                graph_info_tmp = []
                for i in batch_graph_info:
                    graph_info_tmp.append(i[key].numpy())
                graph_info_tmp = np.array(graph_info_tmp)
                graph_info_tmp = torch.tensor(graph_info_tmp)
                batch_labels[key] = graph_info_tmp
        save_graphs(os.path.join(self.data_config['dataset_path'], 'all_graphs_' + str(batch_num) + '.bin'), batch_g, batch_labels)


    def read_all_graphs(self): # prop_per_node=False Deprecated!
        if self.data_config['load_from_binary']:
            try:
                graph_path = os.readlink(os.path.join(self.data_config['dataset_path'], 'all_graphs.bin'))
            except:
                graph_path = 'all_graphs.bin'
            cwd = os.getcwd()
            os.chdir(self.data_config['dataset_path'])
            graph_list, graph_labels = load_graphs(graph_path)
            os.chdir(cwd)
        else:
            num_graph_per_core = self.number_of_graphs // self.data_config['num_of_cores'] + 1
            graph_index        = [x for x in range(self.number_of_graphs)]
            batch_index        = [graph_index[x: x + num_graph_per_core] for x in range(0, self.number_of_graphs, num_graph_per_core)]
            processes = []

            print('Waiting for all subprocesses...')
            for batch_num, batch_index_list in enumerate(batch_index):
                p = multiprocessing.Process(target=self.read_batch_graphs, args=[batch_index_list, batch_num])
                p.start()
                processes.append(p)
            print(processes)
            for process in processes:
                process.join()
            print('All subprocesses done.')
            graph_list = []
            graph_labels = {}
            for x in range(self.data_config['num_of_cores']):
                batch_g, batch_labels = load_graphs(os.path.join(self.data_config['dataset_path'],
                                                                 'all_graphs_' + str(x) + '.bin'))
                graph_list.extend(batch_g)
                for key in batch_labels.keys():
                    try:
                        # graph_labels[key].extend(batch_labels[key])
                        graph_labels[key] = torch.cat([graph_labels[key], batch_labels[key]], 0)
                    except KeyError:
                        graph_labels[key] = batch_labels[key]

                os.remove(os.path.join(self.data_config['dataset_path'], 'all_graphs_' + str(x) + '.bin'))
            save_graphs(os.path.join(self.data_config['dataset_path'], 'all_graphs.bin'), graph_list, graph_labels)

            with open(os.path.join(self.data_config['dataset_path'], 'graph_build_scheme.json'), 'w') as fjson:
                json.dump(self.data_config, fjson, indent=4)
        return graph_list, graph_labels


class ExtractVaspFiles(object):
    """ Extract the force and energy from VASP Files, such as CONTCAR and OUTCAR,
    force is the property of every atom, energy is the property of the molecular. """
    def __init__(self, **data_config):
        self.data_config = {**default_data_config, **config_parser(data_config)}
        if not os.path.exists(self.data_config['dataset_path']):
            os.makedirs(self.data_config['dataset_path'])
        self.in_path_list = np.loadtxt(self.data_config['path_file'], dtype=str)    # path.log
        self.batch_index = np.array_split([x for x in range(len(self.in_path_list))], self.data_config['num_of_cores'])
        self.working_dir = os.getcwd()

    def split_output(self, process_index):
        """ Process the path index """
        f_csv = open(os.path.join(self.data_config['dataset_path'], f'fname_prop_{process_index}.csv'), 'w', buffering=1)
        in_path_index = self.batch_index[process_index]
        for path_index in tqdm(in_path_index, desc='Extracting ' + str(process_index) + ' VASP files.', delay=process_index): # tqdm(batch_fname, desc='Reading ' + str(batch_num) + ' batch graphs', delay=batch_num)
            in_path = self.in_path_list[path_index]
            in_path = in_path.strip("\n")
            os.chdir(in_path)

            if os.path.exists('OUTCAR') and os.path.exists('XDATCAR'):
                read_good = True
                try:
                    # read frames
                    frames_xdatcar  = read('XDATCAR', index=slice(0, None, 5))
                    frames_outcar  = read('OUTCAR', index=slice(0, None, 5))   # coordinates in OUTCAR file are less accurate than that in XDATCAR. Energy in OUTCAR file is more accurate than that in OSZICAR file

                    # pre processing
                    free_energy = [x.get_total_energy() for x in frames_outcar]
                    num_atoms = len(frames_outcar[0])
                    num_frames = len(frames_outcar)

                    # check magnetic moments (magmom).
                    if self.data_config['mask_reversed_magnetic_moments']:
                        frames_outcar[0].get_magnetic_moments()

                    assert len(frames_outcar) ==  num_frames, f'Inconsistent number of frames between OUTCAR and XDATCAR files. OUTCAR: {len(frames_outcar)}; XDATCAR {len(frames_xdatcar)}'
                    assert num_frames == len(free_energy), 'Number of frams does not equal to number of free energies.'

                except:
                    print(f'Read OUTCAR, and/or CONTCAR with exception in: {in_path}')
                    read_good = False

                if read_good:
                    output_mask = [True for x in range(num_frames)]
                    report_mask = False

                    # check similar frames
                    if self.data_config['mask_similar_frames']:
                        no_mask_list = [0]
                        for i, e in enumerate(free_energy):
                            if abs(e - free_energy[no_mask_list[-1]]) > self.data_config['energy_stride']:
                                no_mask_list.append(i)
                        if not no_mask_list[-1] == num_frames - 1: # keep the last frame
                            no_mask_list.append(num_frames - 1)
                        output_mask = [x if i in no_mask_list else False for i,x in enumerate(output_mask)]

                    # # check electronic steps
                    # for i, outcar in enumerate(frames_outcar):
                    #     if ee_steps[i] >= NELM:
                    #         output_mask[i] = False
                    #         report_mask = True

                    # # check magnetic moments
                    # if self.data_config['mask_reversed_magnetic_moments']:
                    #     for i, outcar in enumerate(frames_outcar):
                    #         magmoms = outcar.get_magnetic_moments()
                    #         print(i)
                    #         print(magmoms)
                    #         if not (magmoms > self.data_config['mask_reversed_magnetic_moments']).all():
                    #             output_mask[i] = False
                    #             report_mask = True

                    no_mask_list   = [i for i,x in enumerate(output_mask) if x]
                    free_energy    = [free_energy[x] for x in no_mask_list]
                    frames_outcar  = [frames_outcar[x] for x in no_mask_list]
                    free_energy_per_atom = [x / num_atoms for x in free_energy]

                    # save frames
                    for i in range(len(no_mask_list)):
                        fname = str(os.path.join(self.data_config['dataset_path'], f'POSCAR_{process_index}_{path_index}_{i}'))
                        while os.path.exists(os.path.join(self.working_dir, fname)):
                            fname = fname + '_new'

                        frames_xdatcar[i].write(os.path.join(self.working_dir, fname))
                        forces = frames_outcar[i].get_forces(apply_constraint=False)
                        # stress = frames_outcar[i].get_stress()
                        np.save(os.path.join(self.working_dir, f'{fname}_force.npy'), forces)
                        np.save(os.path.join(self.working_dir,f'{fname}_energy.npy'), free_energy_per_atom[i])
                        # np.save(os.path.join(self.working_dir,f'{fname}_stress.npy'), stress)
                        f_csv.write(os.path.basename(fname) + ',  ')
                        f_csv.write(str(free_energy_per_atom[i]) + ',  ' + str(in_path) + '\n')

                    if report_mask:
                        print(f'Frame(s) in {in_path} are masked.')
            else:
                print(f'OUTCAR, CONTCAR files do not exist in {in_path}.')

            os.chdir(self.working_dir)
        f_csv.close()

    def __call__(self):
        """ The __call__ function """

        processes = []
        for process_index in range(self.data_config['num_of_cores']):
            p = multiprocessing.Process(target=self.split_output, args=[process_index,])
            p.start()
            processes.append(p)
        print(processes)

        for process in processes:
            process.join()

        f = open(os.path.join(self.working_dir, self.data_config['dataset_path'], 'fname_prop.csv'), 'w')
        for job in range(self.data_config['num_of_cores']):
            lines = np.loadtxt(os.path.join(self.working_dir, self.data_config['dataset_path'], f'fname_prop_{job}.csv'), dtype=str)
            np.savetxt(f, lines, fmt='%s')
        f.close()

class BuildDatabase():
    def __init__(self, **data_config):
        self.data_config = {**default_data_config, **config_parser(data_config)}

    def build(self):
        # extract vasp files.
        evf = ExtractVaspFiles(**self.data_config)()

        # build binary DGL graphs.
        graph_reader = ReadGraphs(**self.data_config)
        graph_reader.read_all_graphs()

        # Select whether to save the procedure file
        if not self.data_config['keep_readable_structural_files']:
            fname_prop_data = np.loadtxt(os.path.join(self.data_config['dataset_path'],
                                                      'fname_prop.csv'),
                                         dtype=str, delimiter=',')
            fname_list      = fname_prop_data[:,0]
            for fname in fname_list:
                os.remove(os.path.join(self.data_config['dataset_path'], fname))
                os.remove(os.path.join(self.data_config['dataset_path'], f'{fname}_energy.npy'))
                os.remove(os.path.join(self.data_config['dataset_path'], f'{fname}_force.npy'))
                # os.remove(os.path.join(self.data_config['dataset_path'], f'{fname}_stress.npy'))
            # os.remove(os.path.join(self.data_config['dataset_path'], 'fname_prop.csv'))
            for i in range(self.data_config['num_of_cores']):
                os.remove(os.path.join(self.data_config['dataset_path'], f'fname_prop_{i}.csv'))


class BuildOneGraph(object):
    """ Read one CONTCAR and return a ``DGL.graph``."""
    """
    .. Hint::
        We encourage you to use ``ase`` module to build crystal graphs.
        The ``pymatgen`` module needs some dependencies that conflict with
        other modules.
    """

    def __init__(self, **data_config):
        self.data_config = {**default_data_config, **config_parser(data_config)}

        # check inputs
        if self.data_config['topology_only']:
            self.data_config['build_properties']['energy'] = False
            self.data_config['build_properties']['forces'] = False
            self.data_config['build_properties']['cell'] = False
            self.data_config['build_properties']['stress'] = False
            self.data_config['build_properties']['frac_coords'] = False

        """ Although we recommend representing atoms with one hot code, you can
        use the another way with the next line: """
        self.all_atom_feat = get_atomic_features(self.data_config['species'])
        # self.all_atom_feat = get_atomic_feature_onehot(self.data_config['species'])

        """ In order to build a reasonable graph, a samll cell should be repeated.
        One can modify "self._cell_length_cutoff" for special needs. """
        self._cell_length_cutoff = np.array([11.0, 5.5, 0.0])  # Unit: angstrom. # increase cutoff to repeat more.

        if self.data_config['mode_of_NN'] == 'voronoi' or self.data_config['mode_of_NN'] == 'pymatgen_dist':
            ''' We encourage you to use ``ase`` module to build crystal graphs.
            The ``pymatgen`` module may need dependencies that conflict with
            other modules.'''
            try:
                from pymatgen.core.structure import Structure
                from agat.data.PymatgenStructureAnalyzer import VoronoiConnectivity
            except ModuleNotFoundError:
                raise ModuleNotFoundError('We encourage you to use ``ase`` module \
                to build crystal graphs. The ``pymatgen`` module needs some \
                dependencies that conflict with other modules. Install\
                ``pymatgen`` with ``pip install pymatgen``.')

            self.Structure = Structure
            self.VoronoiConnectivity = VoronoiConnectivity

        # self.device = self.data_config['device']

        if self.data_config['build_properties']['path']:
            print('You choose to store the path of each graph. '
                  'Be noted that this will expose your file structure when you publish your dataset.')
        self.dtype = torch.float

    def get_ndata(self, crystal):
        """ the atomic representations of a crystal graph. """
        ndata = []
        if isinstance(crystal, ase.Atoms):
            # num_sites = len(crystal)
            for i in crystal.get_chemical_symbols():
                ndata.append(self.all_atom_feat[i])
        else:  # isinstance(crystal, self.Structure)
            num_sites = crystal.num_sites
            for i in range(num_sites):
                ndata.append(self.all_atom_feat[crystal[i].species_string])

        return np.array(ndata)

    def get_graph_from_ase(self, fname):  # ase_atoms or file name
        """ return a bidirectional graph with self-loop connection and a dict of information of graph-level features. """

        assert not isinstance(fname, ase.Atoms) or not self.data_config['build_properties']['forces'], \
            'The input cannot be a ase.Atoms object when include_forces is ``True``, try with an ``OUTCAR``'

        if not isinstance(fname, ase.Atoms):
            ase_atoms = read(fname)
        else:
            ase_atoms = fname
        num_sites = len(ase_atoms)

        if self.data_config['mode_of_NN'] == 'ase_dist':
            assert self.data_config['cutoff'], ('The `cutoff` cannot be `None` '
                                                'in this case. Provide a float here.')
            ase_cutoffs = self.data_config['cutoff']
        elif self.data_config['mode_of_NN'] == 'ase_natural_cutoffs':
            ase_cutoffs = np.array(natural_cutoffs(ase_atoms))

        i, j, d, D = ase.neighborlist.neighbor_list('ijdD',  # i: sender; j: receiver; d: distance; D: direction
                                                    ase_atoms,
                                                    cutoff=ase_cutoffs,
                                                    self_interaction=True)
        ndata = self.get_ndata(ase_atoms)
        bg = dgl.graph((i, j))
        bg.ndata['h'] = torch.tensor(ndata, dtype=self.dtype)
        if self.data_config['build_properties']['distance']:
            bg.edata['dist'] = torch.tensor(d, dtype=self.dtype)
        if self.data_config['build_properties']['direction']:
            bg.edata['direction'] = torch.tensor(D, dtype=self.dtype)
        # if self.data_config['build_properties']['forces']:
        #     forces_true = torch.tensor(np.load(fname + '_force.npy'), dtype=self.dtype)
        #     bg.ndata['forces_true'] = forces_true
        if self.data_config['build_properties']['cart_coords']:
            bg.ndata['cart_coords'] = torch.tensor(ase_atoms.positions, dtype=self.dtype)
        if self.data_config['build_properties']['frac_coords']:
            bg.ndata['frac_coords'] = torch.tensor(ase_atoms.get_scaled_positions(), dtype=self.dtype)

        return bg

    def get_graph(self, crystal_fname):
        """ choose which graph-construction method is used, and return a bidirectional graph with self-loop connection. """
        if self.data_config['mode_of_NN'] == 'ase_natural_cutoffs' or self.data_config['mode_of_NN'] == 'ase_dist':
            return self.get_graph_from_ase(crystal_fname)

    def build(self):
        bg = self.get_graph(self.data_config['path_file'])
        return bg

if __name__ == '__main__':
    database = BuildDatabase(path_file='../data/paths.log', dataset_path='./dataset', num_of_cores=16)
    database.build()
    # pass

