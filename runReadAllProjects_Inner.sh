#!/bin/sh

export PYTHONPATH=/scratch/grassl/python-dependencies/lib/python3.7/site-packages/:$PYTHONPATH
export PATH=/scratch/grassl/python-dependencies/bin:$PATH
export NLTK_DATA=/scratch/grassl/python-dependencies/nltk_data

cd /scratch/grassl/Code/readProjects
python3 readAllProjects.py /scratch/fraedric/scratch/projects $1 
