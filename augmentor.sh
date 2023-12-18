#!/bin/bash
#SBATCH --job-name=image_augmentation
#SBATCH --time=06:30:00        
#SBATCH --output=unzip_output.log
#SBATCH --error=unzip_error.log
#SBATCH--mail-user=anton.malkovski@gmail.com
#SBATCH--mail-type=ALL
#SBATCH --partition=AMD

cd /gpfs/space/home/amlk/data/source

module load python
srun python ./augmentor.py