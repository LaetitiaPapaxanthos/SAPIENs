# SAPIENs

This repository contains the code of the model **S**equence-**A**ctivity **P**rediction **I**n **E**nsemble of **N**etwork**s** (SAPIENs). The provided code allows to reproduce the prediction results presented in the manuscript "Large-scale DNA-based phenotypic recording and deep learning enable highly accurate sequence-function mapping". Further detail about the model can be found in the main document and in the machine learning annex of the manuscript. 

The repository is organised as follows:

1. [**code/trained_model**](code/trained_model) contains the weights of the trained model presented in the main document of the aforementioned paper and a Jupyter notebook that loads the weights of the model and computes the main predictive performance results (Figure 4 of the manuscript).

2. [**code/training**](code/training) contains scripts that allow to entirely retrain the model, given the hyperparameters chosen on the validation set. This directory also contains [**results_sample/**](code/training/results_sample), which shows the output of SAPIENs obtained with the sample training and validation sets provided in [**data/**](data/) ([**sequences_train_sample.npy**](data/sequences_train_sample.npy), [**targets_train_sample.npy**](data/targets_train_sample.npy), [**sequences_validation_sample.npy**](data/sequences_validation_sample.npy), [**targets_validation_sample.npy**](data/targets_validation_sample.npy)).

3. [**data**](data) contains the RBS sequences (as DNA strings) and their RBS strength (target), split across training, validation and test sets. This split is the one used for the main document of the manuscript ("Split0"). The sampled training and validation sets are extracted from the training and validation sets used in the manuscript. 

## Loading the trained model [code/trained_model](code/trained_model)
The trained model can be loaded in the Jupyter notebook [**load_trained_model.ipynb**](code/trained_model/notebook/load_trained_model.ipynb) and can be used to predict the target values of the sequences in the test set. The weights of the model are available in [**weights**](code/trained_model/weights).

## Training the model [code/training](code/training)
The repository contains [**resnet_model.py**](code/training/resnet_model.py) that implements the core architecture of a single resnet model, [**run_resnet.py**](code/training/run_resnet.py) that trains the model given a set of hyperparameters and [**main.sh**](code/training/main.sh) that loads the data, trains the ensemble on five GPUs at a time and saves the prediction results. The hyperparameters are set in the script [**utils_model.py**](code/training/utils_model.py). To train the model, please run *bash main.sh*.
The input file format corresponds to the standard binary file format in Python (*.npy* format), with one DNA sequence per row (no further preprocessing is needed). 

## Dependencies

The code only supports python3 and requires the following packages:
+ numpy (tested on 1.16.3)
+ sklearn (tested on 0.21.3)
+ matplotlib (tested on 3.1.1)
+ tensorflow==1.12.0
+ keras==2.2.4

## Contact
laetitia.papaxanthos@bsse.ethz.ch, laetitia.papaxanthos@gmail.com
