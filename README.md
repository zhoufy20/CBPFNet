<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Times+New+Roman&weight=100&size=30&pause=1000&color=000000&center=true&vCenter=true&repeat=false&width=600&lines=Covalent+Bond+Peak+Force+Prediction+Network" alt="Typing SVG" /></a>

we introduce CBPFNet (Covalent Bond Peak Force Prediction Network), a graph attention neural network model, which successfully simulates the dynamic process of covalent bond cleavage and accurately predicts the stress response of covalent bonds under various mechanical loads. Our model provides a perspective for training molecular datasets using GATs and defining covalent bond strength criteria, providing a powerful and open tool for predicting covalent bond peak force.





## Installation

> conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
>
> conda install -c dglteam/label/th24_cu124 dgl



## Usage

### Process the vasp calculation file

> find . -name OUTCAR > paths.log
>
> sed -i 's/OUTCAR$//g' paths.log
>
> sed -i "s#^.#${PWD}#g" paths.log



### Training CBPFNet



### Reference

If you use  Covalent Bond Peak Force dataset, please cite [this paper]():

```

```



### Development & Bugs



`CBPFNet` is under active development, if you encounter any bugs in installation and usage, please open an [issue](). We appreciate your contributions!
