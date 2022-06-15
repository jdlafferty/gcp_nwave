A cupy implementation of nwave experiment 

## Directory Structure
- nwave/ : implementation of nwave algorithms.
    - dictlearner.py :  update rule of feedforward weights
    - neurodynamics.py : update rule of activations
    - wave.py : combine dictlearner and neurodynamics
    - utils.py : get convolutional kernels
- experiment.py : interface to experiment
- dataloader.py : loading data 
- act_vis.py : visualization tools
- configs.py : store all the hyperparameters
- visualization.py : plot l0, l1 and l2 errors
- Word_activations.ipynb : plot activations according to certain words
- test.py : save receptive fields 


# start 
gcloud compute instances start --zone "us-east4-b" "rob-test"

gcloud compute instances stop --zone "us-east4-b" "rob-test"

gcloud compute ssh --zone "us-east4-b" "rob-test"  --tunnel-through-iap --project "jl2994-nerualnet-project-aa1f"


# Activate Conda 
source /home/tracey/miniconda3/bin/activate tracey