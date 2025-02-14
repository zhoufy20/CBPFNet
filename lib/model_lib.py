# @Time    : 2024/9/24 09:14
# @Author  : Feiyu
# @File    : main.py
# @diligent：What doesn't kill me makes me stronger.
# @Function: Some functions used in model.

#===================================================================================================
#### reference from DOI:https://doi.org/10.1016/j.joule.2023.06.003
#### reference code from url:https://github.com/jzhang-github/AGAT
###  cite:Zhang, Jun & Wang, Chaohui & Huang, Shasha & Xiang, Xuepeng & Xiong, Yaoxu & Biao, Xu
###  & Ma, Shihua & Fu, Haijun & Kai, Jijung & Kang, Xiongwu & Zhao, Shijun. (2023).
###  Design high-entropy electrocatalyst via interpretable deep graph attention learning.
###  Joule. 7. 10.1016/j.joule.2023.06.003.
#===================================================================================================

import os
import numpy as np
from ase.io import read
from ase.data import covalent_radii, atomic_numbers
import json, copy, ase, torch


def save_model(model, model_save_dir='agat_model'):
    """Saving PyTorch model to the disk. Save PyTorch model, including parameters and structure. See: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

    :param model: A PyTorch-based model.
    :type model: PyTorch-based model.
    :param model_save_dir: A directory to store the model, defaults to 'agat_model'
    :type model_save_dir: str, optional
    :output: A file saved to the disk under ``model_save_dir``.
    :outputtype: A file.

    """
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    torch.save(model, os.path.join(model_save_dir, 'agat.pth'))

def load_model(model_save_dir='agat_model', device='cuda'):
    """Loading PyTorch model from the disk.

    :param model_save_dir: A directory to store the model, defaults to 'agat_model'
    :type model_save_dir: str, optional
    :param device: Device for the loaded model, defaults to 'cuda'
    :type device: str, optional
    :return: A PyTorch-based model.
    :rtype: PyTorch-based model.

    """
    device = torch.device(device)
    if device.type == 'cuda':
        new_model = torch.load(os.path.join(model_save_dir, 'agat.pth'), weights_only=True)
    elif device.type == 'cpu':
        new_model = torch.load(os.path.join(model_save_dir, 'agat.pth'), weights_only=True,
                               map_location=torch.device(device))
    new_model.eval()
    new_model = new_model.to(device)
    new_model.device = device
    return new_model

def save_state_dict(model, state_dict_save_dir='agat_model', **kwargs):
    """Saving state dict (model weigths and other input info) to the disk. See: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

    :param model: A PyTorch-based model.
    :type model: PyTorch-based model.
    :param state_dict_save_dir: A directory to store the model state dict (model weigths and other input info), defaults to 'agat_model'
    :type state_dict_save_dir: str, optional
    :param **kwargs: More information you want to save.
    :type **kwargs: kwargs
    :output: A file saved to the disk under ``model_save_dir``.
    :outputtype: A file

    """
    if not os.path.exists(state_dict_save_dir):
        os.makedirs(state_dict_save_dir)
    checkpoint_dict = {**{'model_state_dict': model.state_dict()}, **kwargs}
    torch.save(checkpoint_dict, os.path.join(state_dict_save_dir, 'agat_state_dict.pth'))

def load_state_dict(state_dict_save_dir='agat_model'):
    """Loading state dict (model weigths and other info) from the disk. See: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

    :param state_dict_save_dir:  A directory to store the model state dict (model weigths and other info), defaults to 'agat_model'
    :type state_dict_save_dir: str, optional
    :return: State dict.
    :rtype: TYPE

    .. note::
        Reconstruct a model/optimizer before using the loaded state dict.

        Example::

            model = PotentialModel(...)
            model.load_state_dict(checkpoint['model_state_dict'])
            new_model.eval()
            model = model.to(device)
            model.device = device
            optimizer = ...
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    """

    checkpoint_dict = torch.load(os.path.join(state_dict_save_dir, 'agat_state_dict.pth'), weights_only=True)
    return checkpoint_dict

def config_parser(config):
    """Parse the input configurations/settings.

    :param config: configurations
    :type config: str/dict. if str, load from the json file.
    :raises TypeError: DESCRIPTION
    :return: TypeError('Wrong configuration type.')
    :rtype: TypeError

    """

    if isinstance(config, dict):
        return config
    elif isinstance(config, str):
        with open(config, 'r') as config_f:
            return json.load(config_f)
    elif isinstance(config, type(None)):
        return {}
    else:
        raise TypeError('Wrong configuration type.')

class EarlyStopping:
    def __init__(self, model, logger, patience=10, model_save_dir='model_save_dir'):
        """Stop training when model performance stop improving after some steps.

        :param model: AGAT model
        :type model: torch.nn
        :param logger: I/O file
        :type logger: _io.TextIOWrapper
        :param patience: Stop patience, defaults to 10
        :type patience: int, optional
        :param model_save_dir: A directory to save the model, defaults to 'model_save_dir'
        :type model_save_dir: str, optional


        """

        self.model      = model
        self.patience   = patience
        self.counter    = 0
        self.best_score = None
        self.update     = None
        self.early_stop = False
        self.logger     = logger
        self.model_save_dir = model_save_dir

        if not os.path.exists(self.model_save_dir):
            os.makedirs(model_save_dir)
        self.save_model_info()

    def step(self, score, epoch, model, optimizer):
        if self.best_score is None:
            self.best_score = score
            self.update = True
            # self.save_model(model, model_save_dir=self.model_save_dir)
            # self.save_checkpoint(model, model_save_dir=self.model_save_dir)
        elif score > self.best_score:
            self.update = False
            self.counter += 1
            print(f'User log: EarlyStopping counter: {self.counter} out of {self.patience}',
                  file=self.logger)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.update     = True
            self.save_model(model)
            save_state_dict(model, state_dict_save_dir=self.model_save_dir,
                            optimizer_state_dict=optimizer.state_dict(),
                            epoch=epoch, total_loss=score)
            self.counter = 0
        return self.early_stop

    def save_model(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model, os.path.join(self.model_save_dir, 'agat.pth'))
        print(f'User info: Save model with the best score: {self.best_score}',
              file=self.logger)

    # def save_checkpoint(self, model):
    #     '''Saves model when validation loss decrease.'''
    #     torch.save({model.state_dict()}, os.path.join(self.model_save_dir, 'agat_state_dict.pth'))

    def save_model_info(self):
        info = copy.deepcopy(self.model.__dict__)
        info = {k:v for k,v in info.items() if isinstance(v, (str, list, int, float))}
        with open(os.path.join(self.model_save_dir, 'agat_model.json'), 'w') as f:
            json.dump(info, f, indent=4)

def load_graph_build_method(path):
    """ Load graph building scheme. This file is normally saved when you build your dataset.

    :param path: Directory for storing ``graph_build_scheme.json`` file.
    :type path: str
    :return: A dict denotes how to build the graph.
    :rtype: dict

    """

    json_file  = path

    assert os.path.exists(json_file), f"{json_file} file dose not exist."
    with open(json_file, 'r') as jsonf:
        graph_build_scheme = json.load(jsonf)
    return graph_build_scheme

def PearsonR(y_true, y_pred):
    """Calculating the Pearson coefficient.

    :param y_true: The first torch.tensor.
    :type y_true: torch.Tensor
    :param y_pred: The second torch.tensor.
    :type y_pred: torch.Tensor
    :return: Pearson coefficient
    :rtype: torch.Tensor

    .. Note::

        It looks like the `torch.jit.script` decorator is not helping in comuputing large `torch.tensor`, see `agat/test/tesor_computation_test.py` for more details.

    """
    ave_y_true = torch.mean(y_true)
    ave_y_pred = torch.mean(y_pred)

    y_true_diff = y_true - ave_y_true
    y_pred_diff = y_pred - ave_y_pred

    above = torch.sum(torch.mul(y_true_diff, y_pred_diff))
    below = torch.mul(torch.sqrt(torch.sum(torch.square(y_true_diff))),
                             torch.sqrt(torch.sum(torch.square(y_pred_diff))))
    return torch.divide(above, below)


def findinistru(file_path):
    """return a path_list about the initial contcar, which end with /0/"""
    inipaths = []
    with open(file_path, 'r') as file:
        for line in file:
            if '/0/' in line:
                inipaths.append(line.strip())
    return inipaths

def findordinalatoms(contcarpath):
    """Find the ordinal atoms, where the fixed atom ends with 'FFF' in CONTCAR."""
    ordinalatoms = []
    start_line = 5
    with open(contcarpath, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            if line_number >= start_line:
                columns = line.split()
                if 'F' in columns and columns.count('F') >= 2:
                    ordinalatoms.append(line_number - 10)
    return ordinalatoms


def findsamesideatom(contcarpath):
    """
    Identify atoms that are on the same side of a bond as a specified atom.

    Args:
        contcarpath (str): The path to the CONTCAR file.

    Returns:
        vectorstre (numpy array): The vector pointing from one ordinal atom to another.
        neiatoms (list): List of indices of atoms on the same side as the first ordinal atom.
    """
    frames_contcar = read(contcarpath, index='-1:')
    atoms = frames_contcar[-1]
    positions = atoms.get_positions()
    elearr = atoms.get_chemical_symbols()
    ordinalatoms = findordinalatoms(contcarpath)
    # Get the atomic radii
    eleradius = [covalent_radii[atomic_numbers[ele]] for ele in elearr]

    # Function to find neighbors covalently bonding with a given atom
    def find_neighbors(atom_index, positions, eleradius, prev_neighbors):
        neighbors_add = []
        for index in range(len(elearr)):
            # Skip atoms already found
            if index not in prev_neighbors:
                dist = np.linalg.norm(positions[atom_index] - positions[index])
                comparecon = (eleradius[atom_index] + eleradius[index]) * 1.15
                if dist < comparecon:
                    neighbors_add.append(index)
                    prev_neighbors.add(index)
        return prev_neighbors, neighbors_add

    # Start with the first ordinal atom
    current_neighbors_adds = [ordinalatoms[0]]

    # Initialize the list of indices that are on the same side as atom 0
    sameside_indices = set(current_neighbors_adds)
    prev_neighbors = set([ordinalatoms[0], ordinalatoms[1]])

    while current_neighbors_adds:
        new_neighbors_adds = []
        for current_neighbors_add in current_neighbors_adds:
            prev_neighbors, found_neighbors = find_neighbors(current_neighbors_add, positions, eleradius,
                                                             prev_neighbors)
            new_neighbors_adds.extend(found_neighbors)
            sameside_indices.update(found_neighbors)

        current_neighbors_adds = new_neighbors_adds  # Update with new neighbors found

    vectorstre = (positions[ordinalatoms[0]] - positions[ordinalatoms[1]])/np.linalg.norm(positions[ordinalatoms[0]] - positions[ordinalatoms[1]])
    neiatoms = list(sameside_indices)
    return vectorstre, neiatoms


def deepmdgeneratecontcar(outpath, contcarpath, calc, whether_gaussian_noise=True, stretch_factor = 0.02, filenums = 30):
    """Return a list of dgl graph, from initial structure, artificial simulation of bond breaking process"""
    # all the nodes need to move, containing ordinalatoms[0]
    vectorstre, neiatomsside = findsamesideatom(contcarpath)
    frames_contcar = read(contcarpath, index='-1:')
    atoms = frames_contcar[-1]
    os.chdir(outpath)
    ase.io.write('POSCAR', atoms, format='vasp', append='w')

    # calculate the original bond length
    ordinalatoms = findordinalatoms(contcarpath)
    vecneiatoms = atoms.positions[ordinalatoms[0]] - atoms.positions[ordinalatoms[1]]
    orignalbondlength = np.linalg.norm(vecneiatoms)

    bglist, predictionstrainlist = [], []
    dpforcelist = []
    for filenum in range(filenums):
        for neiatom in neiatomsside:
            new_position = [atoms.positions[neiatom][i] + vectorstre[i] * stretch_factor for i in range(3)]
            atoms.positions[neiatom] = new_position
        if whether_gaussian_noise:
            atoms.positions = add_gaussian_noise(atoms.positions, ordinalatoms)
        
        # Dp force 
        atoms.calc = calc
        forces_dp = atoms.get_forces(apply_constraint=False)
        dpforcelist.append(np.linalg.norm(forces_dp))

        vecneiatoms = atoms.positions[ordinalatoms[0]] - atoms.positions[ordinalatoms[1]]
        distance = np.linalg.norm(vecneiatoms)
        strain = distance / orignalbondlength - 1
        predictionstrainlist.append(strain)

        poscar_filename = f"POSCAR{filenum}"
        ase.io.write(poscar_filename, atoms, format='vasp', append='w')

        from data.build_dataset import BuildOneGraph
        database = BuildOneGraph(path_file=poscar_filename, num_of_cores=1)
        bg = database.build()
        bglist.append(bg)
    return bglist, ordinalatoms, predictionstrainlist, dpforcelist


def generatecontcar(outpath, contcarpath, whether_gaussian_noise=True, stretch_factor = 0.02, filenums = 30):
    """Return a list of dgl graph, from initial structure, artificial simulation of bond breaking process"""
    # all the nodes need to move, containing ordinalatoms[0]
    vectorstre, neiatomsside = findsamesideatom(contcarpath)
    frames_contcar = read(contcarpath, index='-1:')
    atoms = frames_contcar[-1]
    os.chdir(outpath)
    ase.io.write('POSCAR', atoms, format='vasp', append='w')

    # calculate the original bond length
    ordinalatoms = findordinalatoms(contcarpath)
    vecneiatoms = atoms.positions[ordinalatoms[0]] - atoms.positions[ordinalatoms[1]]
    orignalbondlength = np.linalg.norm(vecneiatoms)

    bglist, predictionstrainlist = [], []
    for filenum in range(filenums):
        for neiatom in neiatomsside:
            new_position = [atoms.positions[neiatom][i] + vectorstre[i] * stretch_factor for i in range(3)]
            atoms.positions[neiatom] = new_position
        if whether_gaussian_noise:
            atoms.positions = add_gaussian_noise(atoms.positions, ordinalatoms)
        
        vecneiatoms = atoms.positions[ordinalatoms[0]] - atoms.positions[ordinalatoms[1]]
        distance = np.linalg.norm(vecneiatoms)
        strain = distance / orignalbondlength - 1
        predictionstrainlist.append(strain)

        poscar_filename = f"POSCAR{filenum}"
        ase.io.write(poscar_filename, atoms, format='vasp', append='w')

        from data.build_dataset import BuildOneGraph
        database = BuildOneGraph(path_file=poscar_filename, num_of_cores=1)
        bg = database.build()
        bglist.append(bg)
    return bglist, ordinalatoms, predictionstrainlist


def add_gaussian_noise(atom_coordinates, ordinalatoms, noise_std=0.001):
    """
    Gaussian noise with zero mean for each atomic coordinate

    Hint:
        - atom_coordinates: numpy array of atomic coordinates with the shape (n_atoms, 3)
        - noise_std: indicates the standard deviation of noise. The default value is 0.05A """
    noise = np.random.normal(loc=0.0, scale=noise_std, size=atom_coordinates.shape)

    for i in range(atom_coordinates.shape[0]):
        if i in ordinalatoms:
            noise[i, :] = 0
    noisy_coordinates = atom_coordinates + noise
    return noisy_coordinates


def eneforlenoutput(inipaths):
    """output the true energy-force-length
    Hint: now the py is in CH2CHCC2H5/CH2CHCC2H5_2/X
    """
    dirs = os.listdir(inipaths)
    dirs = [d for d in dirs if os.path.isdir(os.path.join(inipaths, d))]
    sorted_dirs = sorted(dirs, key=int)

    fold_num = sum([os.path.isdir(os.path.join(inipaths, dir_name)) for dir_name in sorted_dirs])
    eneforlenarray = np.zeros((4, fold_num))
    truestrainlist = []
    for dir_name in sorted_dirs:
        dir = os.path.join(inipaths, dir_name)
        if os.path.isdir(dir) and dir != '.idea':
            # read the files and return the ordinal (atom1, atom2)
            outcarpath = os.path.join(dir, "OUTCAR")
            contcarpath = os.path.join(dir, "CONTCAR")
            frames_outcar = read(outcarpath, index='-1:')
            frames_contcar = read(contcarpath, index='-1:')
            ordinalatoms = findordinalatoms(contcarpath)

            # read the energy
            num_atoms = len(frames_outcar[0])
            free_energy = [x.get_total_energy() for x in frames_outcar]
            free_energy_per_atom = free_energy[0] / num_atoms
            eneforlenarray[0, int(dir_name)] = free_energy_per_atom

            # read the force
            forces = frames_outcar[0].get_forces(apply_constraint=False)
            force = np.linalg.norm(forces[ordinalatoms[0], :])
            eneforlenarray[1, int(dir_name)] = force

            # read the length
            for i, atoms in enumerate(frames_contcar):
                positions = atoms.get_positions()
                position1 = np.array(positions[ordinalatoms[0], :], dtype=float)
                position2 = np.array(positions[ordinalatoms[1], :], dtype=float)
                distance = np.linalg.norm(position1 - position2)
                eneforlenarray[2, int(dir_name)] = distance
            truestrainlist.append(eneforlenarray[2, int(dir_name)]/eneforlenarray[2, 0]-1)
    return eneforlenarray, truestrainlist




