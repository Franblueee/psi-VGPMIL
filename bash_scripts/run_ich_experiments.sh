#!/bin/bash

#nohup ./bash_scripts/run_ich_experiments.sh &

# rm -f output/*

################################################################################################# 

MAX_ITERS=50

num_feat_array=( 8 32 128 )
num_inducing_array=( 50 100 200 )
kernel_ls_array=( d )
alpha_array=( 1.0 )
beta_array=( 1.0 2.5 4.0 )

for f in "${num_feat_array[@]}"
do
    for m in "${num_inducing_array[@]}"
    do
        for l in "${kernel_ls_array[@]}"
        do
            python code/run_experiment.py --update_hyperparams --save_results --dataset_name="ich" --num_feat=$f --kernel_ls=$l --num_inducing=$m --max_iters=$MAX_ITERS --model_name="VGPMIL" > output/salida_${BASHPID}.txt 2>&1
        done
    done
done

for f in "${num_feat_array[@]}"
do
    for m in "${num_inducing_array[@]}"
    do
        for l in "${kernel_ls_array[@]}"
        do
            for alpha in "${alpha_array[@]}"
            do
                for beta in "${beta_array[@]}"
                do
                    python code/run_experiment.py --update_hyperparams --save_results --dataset_name="ich" --num_feat=$f --kernel_ls=$l --num_inducing=$m --alpha=$alpha --beta=$beta --max_iters=$MAX_ITERS --model_name="G_VGPMIL" > output/salida_${BASHPID}.txt 2>&1
                done
            done
        done
    done
done
