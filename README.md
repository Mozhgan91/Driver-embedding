# Driver Embeddings
This is the code for extracting the driver embeddings for the purpose of driver identification 

# Requirements
All conducted Experiments are done using the following packages: 
* PyTorch v0.4.1 
* Scikit-learn v0.20.0
* math 
* Scipy v1.1.0
* Numpy v1.15.2
* python-weka-wrapper3 v0.1.6
* Matplotlib v3.0.0
* jupyter 
* Orange v3.18.0
* Pandas v0.23.4

# Datasets

The dataset employed in our code be obtained from the following links. 

The HCRL dataset: https://ocslab.hksecurity.net/Datasets/driving-dataset

# Files

## Extraction

* losses folder: implements the triplet loss in the cases of a training set with all time series of the same length, and a training set with time series of unequal lengths;

* networks folder: implements encoder and its building blocks (dilated convolutions, causal CNN);
* scikit_wrappers.py file: implements classes inheriting Scikit-learn classifiers that wrap an encoder and a SVM classifier.
* utils.py file: implements custom PyTorch datasets;
* default_hyperparameters.json file: example of a JSON file containing the hyperparameters of a pair (encoder, classifier).
* sparse_labeling.ipynb file: file containing code to reproduce the results of training an SVM on our representations for different numbers of available labels;
* HouseholdPowerConsumption.ipynb file: Jupyter notebook containing experiments on the Individual Household Electric Power Consumption dataset.
