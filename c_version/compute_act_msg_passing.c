#include <stdio.h>
#include <math.h>
#include <string.h>
#include "matrix_float.c"
#include "read_csv.c"

////////////
/// These functions can be precomputed:
void print_int_matrix(int row, int col, int** m) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%d ", m[i][j]);
        }
        printf("\n");
    }
}

int get_num_nbs(int r){
    int count = 0;
    for (int i = 0; i < (r+1)*(r+1); i++){
        int xi = i/(r+1);
        int yi = i%(r+1);
        int distsq = xi * xi + yi * yi;
        if (distsq <= r*r ){
            count += 1;
        }
    }

    int num_nbs = (count - (r + 1)) * 4 + 1;
    return num_nbs;
}

int* get_struc(int r){
    int rsq = r*r;
    int* l = malloc(sizeof(int) * (2*r + 1));
    for (int i = 0; i < 2*r+1; i++){
        int count  = 0;
        for (int j =0; j < 2*r+1; j++){
            int distsq = (i - r) * (i - r) + (j - r) * (j - r);
            if (distsq <= rsq){
                count+=1;
            }
        }
        l[i] = count;
    }
    return l;
}

int get_index_from_position(int xi, int yi, int xj, int yj, int r){
    int* l = get_struc(r);
    int core_index = (get_num_nbs(r) - 1)/2;
    if (yi == yj){
        int index = core_index + (xj - xi);
        free(l);
        return index;
    }
    else{
        int diff = 0;
        for (int i = 0; i < abs(yj - yi)-1; i++){
            diff += l[r-i-1];
        }
        if (yi > yj){
            int index = core_index - (l[r]-1)/2 - diff - (l[r-(yi - yj)]+1)/2 + (xj - xi);
            free(l);
            return index;
        }
        else if (yi < yj){
            int index = core_index + (l[r]-1)/2 + diff + (l[r-(yi - yj)]+1)/2 + (xj - xi);
            free(l);
            return index;
        }
    }
}

int ** compute_indexset(r, num_nbs, neuron_shape){
    int side_length = sqrt(neuron_shape);
    int** set = malloc(sizeof(int*) * neuron_shape);
    for (int i = 0; i < neuron_shape; i++) {
        set[i] = malloc(sizeof(float) * num_nbs);
    }
    for (int i = 0; i < neuron_shape; i++){
        for (int j = 0 ; j < num_nbs; j++){
            set[i][j] = neuron_shape;
        }
    }
    for (int i = 0; i < neuron_shape; i++){
        int xi = i / side_length;
        int yi = i % side_length;
        for (int j = 0; j < neuron_shape; j++){
            int xj = j / side_length;
            int yj = j % side_length;
            int distsq = (xi - xj)*(xi - xj) + (yi - yj)*(yi - yj);
            if (distsq <= r*r){
                int index = get_index_from_position(xi, yi, xj, yj, r);
                set[i][index] = j;
            }
        }
    }

    return set;
}

float* compute_W(num_nbs, r, w, sigmaE){
    float* W = malloc(sizeof(float) * num_nbs);
    int count = 0;
    for (int i = 0; i < (2*r+1)*(2*r+1); i++){
        int xi = i / (2*r + 1);
        int yi = i % (2*r + 1);
        int distsq = (xi - r)* (xi - r) + (yi - r)* (yi - r);
        if (distsq <= r*r){
            W[count] = exp(- distsq/2.0/sigmaE);
            count += 1;
        }
    }

    float W_E_sum = sum(num_nbs, W);
    for (int i = 0; i < num_nbs; i++){
        W[i] = w * W[i] / W_E_sum;
    }

    return W;
}
///////////

float max(float a, float b){
    if (a >= b){
        return a;
    }
    else{
        return b;
    }
}

float** stimulate(int neuron_shape, int bs, float lr_act, float threshold, float eps, float** stimulus, float** delta_a_exc,
                  float** delta_a_inh, float** exc_act_dummy, float** inh_act_dummy, int leaky,
                  int num_E_nbs, int num_I_nbs, float* W_E, float* W_I, int** N_E, int** N_I){

    //float relative_error;

    for (int t = 0; t < 50; t++) {

        //float **exc_tm1 = copy_matrix(bs, neuron_shape+1, exc_act_dummy);

        // Update of activations
        for (int k = 0; k < bs; k++) {
            for (int i = 0; i < neuron_shape; i++) {

                //Update of exhibitory neurons;
                delta_a_exc[k][i] = - leaky * exc_act_dummy[k][i];
                for (int j = 0; j < num_E_nbs; j++) {
                    delta_a_exc[k][i] += W_E[j] * exc_act_dummy[k][N_E[i][j]];
                }
                for (int j = 0; j < num_I_nbs; j++) {
                    delta_a_exc[k][i] -= W_I[j] * inh_act_dummy[k][N_I[i][j]];
                }
                delta_a_exc[k][i] += stimulus[k][i];
                delta_a_exc[k][i] = lr_act * delta_a_exc[k][i];
                exc_act_dummy[k][i] = exc_act_dummy[k][i] + delta_a_exc[k][i];
                exc_act_dummy[k][i] = max(exc_act_dummy[k][i] - threshold, 0.0) - max(-exc_act_dummy[k][i] - threshold, 0.0);

                //Update of inhibitory neurons;
                delta_a_inh[k][i] = - leaky * inh_act_dummy[k][i];
                for (int j = 0; j < num_E_nbs; j++){
                    delta_a_inh[k][i] += W_E[j] * exc_act_dummy[k][N_E[i][j]];}
                delta_a_inh[k][i] = lr_act * delta_a_inh[k][i];
                inh_act_dummy[k][i] = inh_act_dummy[k][i] + delta_a_inh[k][i];
                inh_act_dummy[k][i] = max(inh_act_dummy[k][i] - threshold, 0.0) - max(-inh_act_dummy[k][i] - threshold, 0.0);

            }
        }

//            float **da = matrix_minus(bs, neuron_shape+1, exc_act_dummy, exc_tm1);
//
//            float sqrt_da = 0;
//            for (int i = 0; i < bs; i++) {
//                for (int j = 0; j < neuron_shape; j++) {
//                    sqrt_da += da[i][j] * da[i][j];
//                }
//            }
//            sqrt_da = sqrt(sqrt_da);
//
//            float sqrt_exc_tm1 = 0;
//            for (int i = 0; i < bs; i++) {
//                for (int j = 0; j < neuron_shape; j++) {
//                    sqrt_exc_tm1 += exc_tm1[i][j] * exc_tm1[i][j];
//                }
//            }
//            sqrt_exc_tm1 = sqrt(sqrt_exc_tm1);
//
//            relative_error = sqrt_da / (eps + sqrt_exc_tm1);
//
//            free_matrix(bs, da);
//
//            free_matrix(bs, exc_tm1);
        }
    return exc_act_dummy;

//        if (relative_error < eps) {
//            printf("relative_error = %f\n", relative_error);
//            return exc_act_dummy;
//        } else {
//            printf("relative_error = %f\n", relative_error);
//            printf("Update doesn't converge.");
//            return exc_act_dummy;
//        }

}


int main() {

    // Hyperparameters
    int ri = 5;
    int re = 3;
    int wi = 5;
    int we = 30;
    int leaky = wi + we;
    int neuron_shape = 1600;
    int sigmaE = 3;
    int bs = 256;
    int imbed_dim = 97;
    float lr_act = 0.01;
    float threshold = 0.01;
    float eps = 5e-3;

    // precomputed parameters
    int num_E_nbs = get_num_nbs(re);
    int num_I_nbs = get_num_nbs(ri);

    int** N_E = compute_indexset(re, num_E_nbs, neuron_shape);
    int** N_I = compute_indexset(ri, num_I_nbs, neuron_shape);

    float* W_E = compute_W(num_E_nbs, re, we, sigmaE);
    float* W_I = compute_W(num_I_nbs, ri, wi, sigmaE);

    // Read Files, and stimulus can also be precomputed
    float** mat = read_matrix(55529, imbed_dim, "word_embeddings.csv");
    float** word_batch = sample_matrix1(55529, imbed_dim, bs, mat);

    float** Phi = read_matrix(97, 1600, "codebook.csv");

    float** stimulus = multiply(bs, imbed_dim, neuron_shape, word_batch, Phi);

    // Malloc activations
    float** exc_act_dummy = malloc_matrix(bs, neuron_shape + 1);
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < neuron_shape + 1; j++) {
            exc_act_dummy[i][j] = 0;
        }
    }

    float** inh_act_dummy = malloc_matrix(bs, neuron_shape + 1);
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < neuron_shape + 1; j++) {
            inh_act_dummy[i][j] = 0;
        }
    }

    float** delta_a_exc = malloc_matrix(bs, neuron_shape); //These are medium variables in the update formula
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < neuron_shape; j++) {
            delta_a_exc[i][j] = 0;
        }
    }

    float** delta_a_inh = malloc_matrix(bs, neuron_shape); //These are medium variables in the update formula
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < neuron_shape; j++) {
            delta_a_inh[i][j] = 0;
        }
    }

    // Update of activations
    exc_act_dummy = stimulate(neuron_shape, bs, lr_act, threshold, eps, stimulus, delta_a_exc, delta_a_inh,
                              exc_act_dummy, inh_act_dummy, leaky, num_E_nbs, num_I_nbs, W_E, W_I, N_E, N_I);

    //print_matrix(bs, neuron_shape, exc_act_dummy);

    write_matrix(bs, neuron_shape, exc_act_dummy, "exc_act.csv");

    return 0;
}

