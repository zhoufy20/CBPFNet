#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :atomic_feature.py
# @Time      :2024/3/24 13:51
# @Author    :Feiyu
# @Main      :On the basis of source code,add another implementation of atomic feature matrix about covalent_radii
#===================================================================================================
#### reference from DOI:https://doi.org/10.1016/j.joule.2023.06.003
#### reference code from url:https://github.com/jzhang-github/AGAT
###  cite:Zhang, Jun & Wang, Chaohui & Huang, Shasha & Xiang, Xuepeng & Xiong, Yaoxu & Biao, Xu
###  & Ma, Shihua & Fu, Haijun & Kai, Jijung & Kang, Xiongwu & Zhao, Shijun. (2023).
###  Design high-entropy electrocatalyst via interpretable deep graph attention learning.
###  Joule. 7. 10.1016/j.joule.2023.06.003.
#===================================================================================================

import numpy as np
from ase.data import atomic_numbers, atomic_masses, covalent_radii


def get_atomic_feature_onehot(Elements_used):
    """Returns a dictionary where keys are element symbols
    and values are corresponding one-hot encoded vectors"""

    number_of_elements = len(Elements_used)
    atomic_number = [atomic_numbers[x] for x in Elements_used]
    atoms_dict = dict(zip(atomic_number, Elements_used))

    atomic_number.sort()

    Elements_sorted = [atoms_dict[x] for x in atomic_number]

    keys               = Elements_sorted
    element_index      = [i for i, ele in enumerate(keys)]
    values             = np.eye(number_of_elements)[element_index]
    atom_feat          = dict(zip(keys, values))
    return atom_feat


def get_atomic_features(Elements_used):
    """Returns a dictionary where keys are element symbols
    and values are corresponding one-hot encoded with atomic mass vectors"""

    number_of_elements = len(Elements_used)

    atomic_number = [atomic_numbers[x] for x in Elements_used]
    atoms_dict = dict(zip(atomic_number, Elements_used))
    atomic_number.sort()
    Elements_sorted = [atoms_dict[x] for x in atomic_number]

    one_hot_matrix = np.zeros((number_of_elements,number_of_elements))

    for idx, ele in enumerate(Elements_sorted):
        one_hot_matrix[idx, idx] = covalent_radii[atomic_numbers[ele]]
    atom_feat = {element: one_hot_matrix[idx] for idx, element in enumerate(Elements_sorted)}
    return atom_feat


if __name__ == '__main__':
    Elements_used = ['Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'B',  'Ba',
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
    atomic_feature = get_atomic_features(Elements_used)
    atomic_feature_onehot = get_atomic_feature_onehot(Elements_used)
    print(atomic_feature)
    # print(atomic_feature_onehot)
    # {'H': array([1., 0., 0., 0.]), 'C': array([0., 1., 0., 0.]),
    # 'N': array([0., 0., 1., 0.]), 'O': array([0., 0., 0., 1.])}

    # {'H': array([1.008, 0., 0., 0.]), 'C': array([0., 12.011, 0., 0.]),
    # 'N': array([0., 0., 14.007, 0.]),'O': array([0., 0., 0., 15.999])}
