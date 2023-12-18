#!/bin/bash
#SBATCH --job-name=image_augmentation
#SBATCH --time=06:30:00        
#SBATCH --output=unzip_output.log
#SBATCH --error=unzip_error.log
#SBATCH--mail-user=anton.malkovski@gmail.com
#SBATCH--mail-type=ALL


cd /gpfs/space/home/amlk/data

module load python
srun python ./augmentor.py