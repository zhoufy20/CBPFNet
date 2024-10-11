<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Times+New+Roman&size=12&pause=1000&color=000000&center=true&repeat=false&width=435&lines=CBPFNet%3A+A+Pretrained+Graph+Attention+Network+for+Covalent+Bond+Loading+Peaks" alt="Typing SVG" /></a>

we introduce **CBPFNet**, a pretrained Covalent Bond Peak Force Network, which simulates the dynamic process of covalent bond cleavage and accurately predicts the stress response under various mechanical loads. This work represents a significant step forward in integrating machine learning and quantum chemistry for bond rupture prediction, offering a powerful and open tool for accelerating the design and analysis of advanced materials in molecular modeling and materials science.


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

`CBPFNet` is under active development, if you encounter any bugs in installation and usage, please open an [issue](https://github.com/zhoufy20/CBPFNet/issues/new)). We appreciate your contributions!
