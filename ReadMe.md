A cupy implementation of the nwave experiments

## Directory Structure
- nwave/ : implementation of nwave algorithms.
    - dictlearner.py :  update rule of feedforward weights
    - neurodynamics.py : update rule of activations
    - wave.py : combine dictlearner and neurodynamics
    - utils.py : store many functions that will be used in other files
- experiment.py : interface to experiment, including training, plotting receptive fields and plotting activations
- dataloader.py : load data 
- configs.py : read all the hyperparameters
- compute_whole_activ.py: compute whole "activity.npy", which contains the activations of over 50,000 words
- compute_act_convolve.py: Don't compute whole activity file. Given some words, plot corresponding activations using spicy.convolve
- compute_act_laplacian.py: Don't compute whole activity file. Given some words, plot corresponding activations using multiplication of laplacian matrix and activations
- matrix.py: implement vector multiplication and matrix multiplication in pure python
- c_matrix/ : convert the program to c
    - makefile: to compile the c code
    - matrix.c: implement vector multiplication and matrix multiplication in pure c


# start 
To run the project locally, run `python experiment.py --row [test number] --processor [CPU/GPU] `, and all the results will be stored in the result folder.

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