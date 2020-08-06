#!/bin/sh
WDIR='/scratch/grassl/out'

sbatch -A anywhere -p anywhere --exclusive -J readAllProjects -o /scratch/grassl/out.log runReadAllProjects_Inner.sh $WDIR
