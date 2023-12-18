#!/bin/bash
#SBATCH --job-name=unzip_job
#SBATCH --time=00:30:00               # Adjust as needed (hours:minutes:seconds)
#SBATCH --output=unzip_output.log
#SBATCH --error=unzip_error.lo

module load python
srun puthon ./extract_hpc.py