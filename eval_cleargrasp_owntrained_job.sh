#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080 # partition (queue)
#SBATCH --mem 128G # memory pool for each core (4GB)
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -c 8 # number of cores
#SBATCH --gres=gpu:2
#SBATCH -D /work/dlclarge1/kaeppelm-cleargrasp
#SBATCH -o %x.%N.%j.out # STDOUT  
# #SBATCH -e %x.%N.%j.err # STDERR
#SBATCH -J cleargrasp_eval_cleargrasp_owntrained # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)

# Print some information about the job to STDOUT
cd /work/dlclarge1/kaeppelm-cleargrasp/
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Job to perform
source ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kaeppelm/depth-via-interaction/cleargrasp/api/depth2depth/gaps/lib/x86_64/
conda activate cleargrasp
cd /home/kaeppelm/depth-via-interaction/cleargrasp/eval_depth_completion
srun python eval_depth_completion.py -c config/config_real_known_owntrained.yaml
srun python eval_depth_completion.py -c config/config_real_novel_owntrained.yaml
srun python eval_depth_completion.py -c config/config_synthetic_known_owntrained.yaml
srun python eval_depth_completion.py -c config/config_synthetic_novel_owntrained.yaml

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
