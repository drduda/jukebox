#!/bin/bash
#SBATCH --job-name=generate_embeddings_dataset
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hpc3_gpu

srun python encoding_dataset_generator.py --size=medium --audio_dir=~/data --output_dir=~/data/embeddings --batch_size=4
