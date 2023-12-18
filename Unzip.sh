#!/bin/bash
#SBATCH --job-name=unzip_job
#SBATCH --time=01:30:00               # Adjust as needed (hours:minutes:seconds)
#SBATCH --output=unzip_output.log
#SBATCH --error=unzip_error.lo

module load python
srun python ./extract_hpc.py