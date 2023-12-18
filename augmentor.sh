#!/bin/bash
#SBATCH --job-name=image_augmentation
#SBATCH --time=06:30:00        
#SBATCH --output=unzip_output.log
#SBATCH --error=unzip_error.log
#SBATCH--mail-user=anton.malkovski@gmail.com
#SBATCH--mail-type=ALL


cd /gpfs/space/home/amlk/data

module load python/3.9.12
module load py-pandas/1.1.4
module load py-numpy/1.19.2
srun python3.9.12 augmentor.py