#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=fc_haeffnerbem
#
# Partition:
#SBATCH --partition=savio_bigmem
#
# Request one node:
#SBATCH --nodes=1
#
# Wall clock limit:
#SBATCH --time=00:10:00
#
## Command(s) to run:
module load python/3.9.12
source activate bem39
ipython run.py > job.pyout
