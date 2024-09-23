<img src="https://readme-typing-svg.demolab.com?font=Times+New+Roman&weight=100&size=30&pause=1000&color=000000&center=true&vCenter=true&repeat=false&width=600&lines=Covalent+Bond+Peak+Force+Prediction+Network" alt="Typing SVG" />

we introduce CBPFNet (Covalent Bond Peak Force Prediction Network), a graph attention neural network model, which successfully simulates the dynamic process of covalent bond cleavage and accurately predicts the stress response of covalent bonds under various mechanical loads. Our model provides a perspective for training molecular datasets using GATs and defining covalent bond strength criteria, providing a powerful and open tool for predicting covalent bond peak force.



## Installation with anaconda

`````
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

conda install -c dglteam/label/th24_cu124 dgl
`````



## Usage

### Process the vasp calculation file

```
find . -name OUTCAR > paths.log
sed -i 's/OUTCAR$//g' paths.log
sed -i "s#^.#${PWD}#g" paths.log
```



### Data processing

`````python
from data.build_dataset import BuildDatabase

if __name__ == '__main__':
	database = BuildDatabase(path_file='..../paths.log', dataset_path="dataset/Dataset")
	database.build()
`````



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
