#!/bin/bash

# Declare path to input, output.
input_path='../../data'
output_path='../../results'

# Declare variables that identify the set of hyperparameters.
models=(1 2)
seed_ensemble=(10 20 30 40 50)
export models_exp=$( IFS=:; printf '%s' "${models[*]}")
export seed_ensemble_exp=$( IFS=:; printf '%s' "${seed_ensemble[*]}")

# Declare which GPU to train the models on.
gpu=(1 2 3 4 5) 

# 1. Preprocess data: DNA (or RNA) sequences to onehot encoding arrays.
python3 data_preprocess.py --input_path ${input_path} --output_path ${input_path}

# 2. Train the ten models of the ensemble. 
for id_model in ${models[*]}
do
for se in ${!seed_ensemble[@]}
do
    python3 run_resnet.py --input_path ${input_path} --output_path ${output_path} \
	--hyperparameters ${id_model} --seed_ensemble ${seed_ensemble[$se]} \
	--save_model True --cuda  ${gpu[$se]}  & # Models are run in parallel
done 
wait  # Wait that training the models for the first set of hyperparameters 
# is done to run models for the second set of hyperparameters.
done
wait

# 3. Print the results in a table.
python3 results_postprocess.py --input_path ${input_path} --model_path ${output_path} --output_path ${output_path}

