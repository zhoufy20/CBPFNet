import argparse
from pymatgen.io.vasp import Poscar
from pymatgen.core import Molecule
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem import rdchem
from mordred import Calculator, descriptors
from itertools import combinations
import numpy as np

def read_contcar(file_path):
    poscar = Poscar.from_file(file_path)
    structure = poscar.structure
    return structure

def get_atom_types(structure):
    return [str(site.species) for site in structure.sites]

def get_lattice_vectors(structure):
    return structure.lattice.matrix

def get_coordinates(structure):
    return structure.cart_coords

def get_covalent_radii():
    # 共价半径（单位：Å），来源：https://en.wikipedia.org/wiki/Covalent_radius
    return {
        "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
        "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39
        # 根据需要添加更多元素
    }

def find_bonds(molecule, tolerance=0.4):
    radii = get_covalent_radii()
    bonds = []
    for atom1, atom2 in combinations(range(len(molecule)), 2):
        elem1 = molecule[atom1].specie.symbol
        elem2 = molecule[atom2].specie.symbol
        r1 = radii.get(elem1, 0.77)  # 默认共价半径
        r2 = radii.get(elem2, 0.77)
        cutoff = r1 + r2 + tolerance
        distance = np.linalg.norm(molecule[atom1].coords - molecule[atom2].coords)
        if distance <= cutoff:
            bonds.append(((atom1, atom2), distance))
    return bonds

def pymatgen_to_rdkit(molecule, bond_lengths):
    """
    将 pymatgen 的 Molecule 对象和键长信息转换为 RDKit 的 Mol 对象。

    :param molecule: pymatgen.core.Molecule 对象
    :param bond_lengths: list of tuples, [((atom1_index, atom2_index), distance), ...]
    :return: RDKit Mol 对象
    """
    # 创建一个可编辑的 RDKit 分子对象
    emol = Chem.RWMol()

    # 添加原子并记录它们在 RDKit 分子中的索引
    atom_idx_map = {}
    for i, site in enumerate(molecule.sites):
        rd_atom = Chem.Atom(site.specie.symbol)
        rd_idx = emol.AddAtom(rd_atom)
        atom_idx_map[i] = rd_idx

    # 添加键
    for bond in bond_lengths:
        (atom1, atom2), distance = bond
        rd_atom1 = atom_idx_map[atom1]
        rd_atom2 = atom_idx_map[atom2]
        # 这里我们假设所有键为单键，你可以根据需要调整
        emol.AddBond(rd_atom1, rd_atom2, Chem.BondType.SINGLE)

    # 返回不可编辑的 Mol 对象
    return emol.GetMol()

def get_bond_orders(mol):
    bond_orders = []
    for bond in mol.GetBonds():
        bond_order = bond.GetBondTypeAsDouble()
        atom1 = bond.GetBeginAtom().GetSymbol()
        atom2 = bond.GetEndAtom().GetSymbol()
        bond_orders.append(((atom1, atom2), bond_order))
    return bond_orders

def calculate_electronegativity_diff(bond_orders):
    electronegativity = {
        "H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
        "P": 2.19, "S": 2.58, "Cl": 3.16, "Br": 2.96, "I": 2.66
    }
    electronegativity_diff = []
    for bond in bond_orders:
        (atom1, atom2), order = bond
        en1 = electronegativity.get(atom1, 2.5)  # 默认值
        en2 = electronegativity.get(atom2, 2.5)
        diff = abs(en1 - en2)
        electronegativity_diff.append(((atom1, atom2), diff))
    return electronegativity_diff

def get_hybridizations(mol):
    return [atom.GetHybridization() for atom in mol.GetAtoms()]

def calculate_polarity(mol):
    return Descriptors.MolLogP(mol)  # 使用 LogP 作为极性近似

def generate_fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

def calculate_mordred_descriptors(mol):
    calc = Calculator(descriptors, ignore_3D=True)
    desc = calc(mol)
    return desc

def main(concar_path):
    # 1. 读取 CONTCAR 文件
    structure = read_contcar(concar_path)

    # 2. 提取描述符
    atom_types = get_atom_types(structure)
    lattice_vectors = get_lattice_vectors(structure)
    coordinates = get_coordinates(structure)

    # 3. 提取键长（基于距离的自定义方法）
    molecule = Molecule([site.specie for site in structure.sites],
                        [site.coords for site in structure.sites])
    bond_lengths = find_bonds(molecule)

    # 4. 转换为 RDKit 分子对象
    try:
        mol = pymatgen_to_rdkit(molecule, bond_lengths)
    except ValueError as e:
        print(e)
        print("无法转换为 RDKit 分子对象，终止程序。")
        return

    # 5. 提取 RDKit 描述符
    bond_orders = get_bond_orders(mol)
    electronegativity_diff = calculate_electronegativity_diff(bond_orders)
    hybridizations = get_hybridizations(mol)
    polarity = calculate_polarity(mol)
    fingerprint = generate_fingerprint(mol)
    mordred_desc = calculate_mordred_descriptors(mol)

    # 6. 输出描述符
    print("=== 原子种类 ===")
    print(atom_types)

    print("\n=== 晶格向量 ===")
    print(lattice_vectors)

    print("\n=== 原子坐标 ===")
    print(coordinates)

    print("\n=== 键长 (前10个) ===")
    for bond, length in bond_lengths[:10]:
        print(f"键: {bond}, 长度: {length:.2f} Å")

    print("\n=== 键级 (前10个) ===")
    for bond, order in bond_orders[:10]:
        print(f"键: {bond}, 键级: {order}")

    print("\n=== 电负性差 (前10个) ===")
    for bond, diff in electronegativity_diff[:10]:
        print(f"键: {bond}, 电负性差: {diff}")

    print("\n=== 杂化状态 (前10个) ===")
    for i, hybrid in enumerate(hybridizations[:10]):
        print(f"原子 {i} 的杂化状态: {hybrid}")

    print("\n=== 分子极性 (LogP) ===")
    print(polarity)

    print("\n=== 分子指纹 (前10位) ===")
    print(list(fingerprint)[:10])

    print("\n=== Mordred 描述符 (前10个) ===")
    print(mordred_desc.head())

    # 7. 绘制分子结构
    img = Draw.MolToImage(mol)
    img.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 CONTCAR 文件中提取分子描述符")
    parser.add_argument("concar", type=str, help="CONTCAR 文件路径")
    args = parser.parse_args()
    main(args.concar)
