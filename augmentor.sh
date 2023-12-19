#!/bin/bash
#SBATCH --job-name=image_augmentation
#SBATCH --time=06:30:00        
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH--mail-user=anton.malkovski@gmail.com
#SBATCH--mail-type=ALL
#SBATCH --partition=intel


cd /gpfs/space/home/amlk/data

module load python

srun python augmentor.py