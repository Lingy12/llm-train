#!/bin/bash -l
#SBATCH --job-name=whisper-tuning   # Job name
#SBATCH --output=log/whisper.o%j # Name of stdout output file
#SBATCH --error=log/whisper.e%j  # Name of stderr error file
#SBATCH --partition=defq  # or ju-standard-g, partition name
#SBATCH --nodes=4               # Total number of nodes 
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=5-00:00:00       # Run time (d-hh:mm:ss)

node_list=($(scontrol show hostnames))
export RDZV_HOST=${node_list[0]}
# export RDZV_HOST=$(hostname -I | awk '{print $1}')
export RDZV_PORT=29400
echo -e "Running on hosts: $(echo $(scontrol show hostname))"
echo "RDZV_HOST=$RDZV_HOST"
echo "RDZV_PORT=$RDZV_PORT"


srun --mpi=pmix \
--container-image /mnt/home/lingy/images/llm_train.sqsh\
--container-writable \
--container-mounts /mnt/home/lingy:/root \
--container-remap-root \
--container-workdir /root/workspace/projects/llm-train \
bash run_training.sh

# srun --mpi=pmix \
# --container-image /mnt/home/lingy/images/my_torch.sqsh \
# --container-writable \
# --container-mounts /mnt/home/lingy:/root \
# --container-remap-root \
# ls /root/workspace
