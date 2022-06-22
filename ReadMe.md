A cupy implementation of nwave experiment 

## Directory Structure
- nwave/ : implementation of nwave algorithms.
    - dictlearner.py :  update rule of feedforward weights
    - neurodynamics.py : update rule of activations
    - wave.py : combine dictlearner and neurodynamics
    - utils.py : store many functions that will be used in other files
- experiment.py : interface to experiment, including training, plotting receptive fields and plotting activations
- dataloader.py : loading data 
- configs.py : store all the hyperparameters


# start 
To run the code, we only need to run experiment.py, and all the results will be stored in the result folder, by using `python experiment.py -- row [test number]`

To run the code on gcloud, using the following command:

`gcloud compute instances start --zone "us-east4-b" "rob-test"`

`gcloud compute instances stop --zone "us-east4-b" "rob-test"`

`gcloud compute ssh --zone "us-east4-b" "rob-test"  --tunnel-through-iap --project "jl2994-nerualnet-project-aa1f"`