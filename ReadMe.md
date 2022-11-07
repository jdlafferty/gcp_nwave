A cupy and FPGA implementation of the Nwave Dynamics Model.

## Directory Structure
- nwave/ : implementation of nwave algorithms.
    - dictlearner.py :  update rule of feedforward weights
    - neurodynamics.py : update rule of activations
    - wave.py : combine dictlearner and neurodynamics
    - utils.py : store many functions that will be used in other files

- c_version/ : convert the program to c
  
    - makefile: to compile the c code
    - compute_act_laplacian.c: compute activations using laplacian matrix
    - compute_act_msg_passing: compute activations using  message passing algorithm
    - compute_RC.c: compute receptive field using trained codebook
    - matrix_float.c: implement vector multiplication and matrix multiplication in pure c
    - msg_passing_algorithm.c: implement message passing algorithm in C
    - plot_RC.ipynb: plot receptive field given data from compute_RC.c 
    - read_csv.c: read data from csv files
    - read_parameter.c: read hyperparameters in csv files
    - test_between_C_and_python.ipynb: compare the results between C version and python version
    - train_laplacian.c: implement training steps using laplacian matrix in C
    - train_msg_passing.c: implement training steps using message passing algorithm in C
    
- FPGA_version/ : convert the program to FPGA

    For ACT language, refer to https://avlsi.csl.yale.edu/act/doku.php?id=language:start.
    - example.act: sample code from Prof.Rajit

- variables_record/ : record and analyze all the values during the computation
    - record_variable_values.py: record all the values. Index are from random sampling the batch; value are from the computations
    - analyze_variable_values.py: analyze and make histogram plots

- experiment.py : interface to experiment, including training, plotting receptive fields and plotting activations

- dataloader.py : load data 

- configs.py : read all the hyperparameters

- compute_whole_activ.py: compute whole "activity.npy", which contains the activations of over 50,000 words
- compute_act_convolve.py: Don't compute whole activity file. Given some words, plot corresponding activations using spicy.convolve
- compute_act_laplacian.py: Don't compute whole activity file. Given some words, plot corresponding activations using multiplication of laplacian matrix and activations
- compute_act_msg_passing.py: Don't compute whole activity file. Given some words, compute and plot activations using message passing algorithm
- matrix.py: implement vector multiplication and matrix multiplication in pure python
- train_convolve.py: standalone python file of training process, and using convolution
- train_laplacian.py: standalone python file of training process, and using laplacian matrix
- train_msg_passing.py: standalone python file of training process, and using message passing algorithm


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