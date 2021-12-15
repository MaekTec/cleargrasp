#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080 # partition (queue)
#SBATCH --mem 128G # memory pool for each core (4GB)
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -c 8 # number of cores
#SBATCH --gres=gpu:4
#SBATCH -D /work/dlclarge1/kaeppelm-cleargrasp
#SBATCH -o %x.%N.%j.out # STDOUT  
# #SBATCH -e %x.%N.%j.err # STDERR 
#SBATCH -J cleargrasp_training_occlusion_boundaries_interaction_data # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)

# Print some information about the job to STDOUT
cd /work/dlclarge1/kaeppelm-cleargrasp/
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Job to perform
source ~/.bashrc
conda activate cleargrasp
cd /home/kaeppelm/depth-via-interaction/cleargrasp/pytorch_networks/occlusion_boundaries
srun python train.py -c config/config_interaction_dataset.yaml

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
