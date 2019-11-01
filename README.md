# SAPIENs

This repository contains the code of the model **S**equence-**A**ctivity **P**rediction **I**n **E**nsemble of **N**etwork**s** (SAPIENs). The provided code allows to reproduce the prediction results presented in the manuscript "Large-Scale Sequence-Function Mapping and Deep Learning for Prediction of Biological Properties", under review. Further detail about the model can be found in the main document and in the machine learning annex of the manuscript. 

The repository is organised as follows:

1. **code/trained_model/** contains the weights of the trained model presented in the main document of the aforementionned paper and a jupyter notebook that loads the weights of the model and computes the main predictive performance results.

2. **code/training/** contains scripts that allow to entirely retrain the model, given the hyperparemeters chosen on the validation set.

3. **data/** contains the RBS sequences (DNA encoding) and their RBS strength (target), split across training, validation and test sets. This split is the one used for the main document of the manuscript ("Split0").

## Loading the trained model (code/trained_model/)
The trained model can be loaded in the jupyter notebook **notebook/load_trained_model.ipynb** and can be used to predict the target values of the sequences in the test set. The weights of the model are available in **weights/**.

## Training the model (code/training/)
The repository contains **resnet_model.py** that implements the core architecture of a single resnet model, **run_resnet.py** that trains the model given a set of hyperparameters and **main.sh** that loads the data, trains the ensemble on five GPUs at a time and saves the prediction results. The hyperparameters are set in the script **utils_models.py**. 
The input file format corresponds to the standard binary file format in Python (.npy format), with one DNA sequence per row  (no further preprocessing is needed). 

## Contact
laetitia.papaxanthos@bsse.ethz.ch
