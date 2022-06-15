A cupy implementation of nwave experiment 

## Directory Structure
- nwave/ : implementation of nwave algorithms.
    - dictlearner.py : dictionary learning algorithms
    - neurodynamics.py : neuron activity simulator
- experiment.py : interface to experiment
- pallet.py : loading data 
- painterbox.py : visulization tools
- utils.py : utility support


# start 
gcloud compute instances start --zone "us-east4-b" "rob-test"
gcloud compute instances stop --zone "us-east4-b" "rob-test"



gcloud compute ssh --zone "us-east4-b" "rob-test"  --tunnel-through-iap --project "jl2994-nerualnet-project-aa1f"


# Activate Conda 
source /home/tracey/miniconda3/bin/activate tracey