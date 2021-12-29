# Driver Embeddings
This is the code for extracting the driver embeddings for the purpose of driver identification used in our works ([Driver Identification Using Vehicular Sensing Data: A Deep Learning Approach](https://ieeexplore.ieee.org/abstract/document/9417463)) and DriverRep (to be published)


# Requirements
All conducted Experiments are done using the following packages: 
* PyTorch v0.4.1 
* Scikit-learn v0.20.0
* math 
* Scipy v1.1.0
* Numpy v1.15.2
* Matplotlib v3.0.0
* jupyter
* Pandas v0.23.4

# Datasets

The dataset employed in our code be obtained from the following links. 

* The HCRL driving dataset: https://ocslab.hksecurity.net/Datasets/driving-dataset
* The HCIlab driving dataset: https://www.hcilab.org/research/hcilab-driving-dataset/

# Files

## Extraction

* losses: the triplet loss implementation for driving time series of different/same lengths.
* Casual CNN: implements encoder and its building blocks (dilated convolutions, causal CNN);
* LSTM: the implementation of LSTM encoder is done for ablation study 
* model_wrappers: implements classes inheriting Scikit-learn classifiers that wrap an encoder and a SVM classifier.
* utils: implements custom PyTorch datasets;
* main: Jupyter notebook containing experiments on the dataset.
