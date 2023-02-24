## About RLBind

A deep convolutional neural network-based model by integrating local and global features, sequence and structure properties is constructed to predict RNA-ligand binding sites.

The benchmark datasets can be found in ./data_cache/, the codes for RLBind are available in ./src. And the results and model are saved in ./results. Furthermore, the demo and corresponding documentation files can be found in ./demo. See our paper for more details.
#### Paper: Wang K, Zhou R, Wu Y, et al. RLBind: a deep learning method to predict RNA–ligand binding sites. Briefings in Bioinformatics, 2023, 24(1), bbac486.

### Requirements
- python 3.7
- cudatoolkit 10.1.243
- cudnn 7.6.0
- pytorch 1.4.0
- numpy 1.16.4
- scikit-learn 0.21.2
- pandas 0.24.2

The easiest way to install the required packages is to create environment with GPU-enabled version:
```bash
conda env create -f environment_gpu.yml
conda activate RLBind_env
```

### Testing the model

```bash
cd ./src/
python predict.py
```
### Re-training your own model for the new dataset
```bash
cd ./src/
python training.py
```
### contact
Kaili Wang: kailiwang@csu.edu.cn
