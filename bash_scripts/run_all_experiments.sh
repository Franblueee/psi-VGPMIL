# nohup bash_scripts/run_all_experiments.sh &

CUDA_VISIBLE_DEVICES=3 bash_scripts/run_mnist_experiments.sh &
CUDA_VISIBLE_DEVICES=3 bash_scripts/run_musk_experiments.sh &
CUDA_VISIBLE_DEVICES=3 bash_scripts/run_ich_experiments.sh &