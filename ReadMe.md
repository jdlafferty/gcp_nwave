A cupy implementation of nwave experiment 

## Directory Structure
- nwave/ : implementation of nwave algorithms.
    - dictlearner.py :  update rule of feedforward weights
    - neurodynamics.py : update rule of activations
    - wave.py : combine dictlearner and neurodynamics
    - utils.py : store many functions that will be used in other files
- experiment.py : interface to experiment, including training, plotting receptive fields and plotting activations
- dataloader.py : load data 
- configs.py : read all the hyperparameters


# start 
To run the project locally, run `python experiment.py -- row [test number]`, and all the results will be stored in the result folder.

To run the project on gcloud, these are the steps to start and stop an instance on GCP:

Install gcloud cli
  https://cloud.google.com/sdk/docs/install

Login to gcloud using netid:
  `gcloud auth login`

Set project:
  `gcloud config set project jl2994-nerualnet-project-aa1f`

Start instance:
  `gcloud compute instances start --zone "us-east4-b" "rob-test"`

SSH to instance:
  `gcloud compute ssh --zone "us-east4-b" "rob-test" --tunnel-through-iap --project "jl2994-nerualnet-project-aa1f"`

Stop instance
  `gcloud compute instances stop --zone "us-east4-b" "rob-test"`