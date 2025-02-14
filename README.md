## CBPFNet: A Pretrained Graph Attention Network for Covalent Bond Loading Peaks


We introduce CBPFNet, a pretrained **C**ovalent **B**ond **P**eak **F**orce **Net**work, designed to simulate the dynamic process of covalent bond cleavage and accurately predict the stress response under various mechanical loads. This innovative model represents a significant advancement in the integration of machine learning with quantum chemistry for bond rupture prediction. CBPFNet leverages deep learning techniques to model and predict the behavior of covalent bonds under different mechanical stresses, enabling a more efficient and precise approach to material design and analysis. This work builds upon existing frameworks such as [AGAT](https://github.com/jzhang-github/AGAT), refining the methodology to enhance performance in predicting bond rupture dynamics. 

## Arichitecture
![CBPFNet](./assets/architecture.jpg)

## Installation with anaconda

`````
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

conda install -c dglteam/label/th24_cu124 dgl
`````



## Usage

### Process the vasp calculation file

```shell
# The directory of all CONTCAR folders(shell)
find . -name OUTCAR > paths.log
sed -i 's/OUTCAR$//g' paths.log
sed -i "s#^.#${PWD}#g" paths.log
```


### Data processing

```python
from data.build_dataset import BuildDatabase

if __name__ == '__main__':
    database = BuildDatabase(path_file='..../paths.log', dataset_path="dataset/Dataset")
    database.build()
```



### model training and test

There are several Instructions for data process ,model training and test.

```python
from models.predict import Test

if __name__ == '__main__':
    # model training
    f = Fit(dataset_path='dataset/Dataset/all_graphs.bin', output_files='out_put/train')
    f.fit()
    
    # model test
    t = Test(path_file='../Dataset/paths.log', output_files='out_put/datasetpredict')
    t.output()
```



### Reference

If you use  the Covalent Bond Peak Force  Network, please cite [this paper]():

```

```



### Development & Bugs

`CBPFNet` is under active development, if you encounter any bugs in installation and usage, please open an [issue](https://github.com/zhoufy20/CBPFNet/issues/new). We appreciate your contributions!
