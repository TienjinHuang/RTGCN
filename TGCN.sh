#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00
#SBATCH --cpus-per-task=10
#SBATCH -o ./results/test_robust1_kl.out


source /home/liuyu/miniconda3/etc/profile.d/conda.sh
conda activate RTGCN

module purge
module load 2021
module load CUDA/11.3.1
#--reinitialize
#--model WideResNet  CE,kl
python main.py --model_name TGCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 64 --loss mse_with_regularizer --settings supervised --gpus 1 --kl_gamma 0.0 --data shenzhen --noise --noise_sever 2 --noise_ratio 0.1 --noise_ratio_node 0.1
