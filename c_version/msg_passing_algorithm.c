#include <stdio.h>
#include <math.h>
#include <string.h>
#include "matrix_float.c"

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
        else {
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

float* compute_WE(int num_nbs, int r, int we, int sigmaE){
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
        W[i] = we * W[i] / W_E_sum;
    }

    return W;
}

float* compute_WI(int num_nbs, int r, int wi){
    float* W = malloc(sizeof(float) * num_nbs);
    int count = 0;
    for (int i = 0; i < (2*r+1)*(2*r+1); i++){
        int xi = i / (2*r + 1);
        int yi = i % (2*r + 1);
        int distsq = (xi - r)* (xi - r) + (yi - r)* (yi - r);
        if (distsq <= r*r){
            W[count] = 1;
            count += 1;
        }
    }

    float W_E_sum = sum(num_nbs, W);
    for (int i = 0; i < num_nbs; i++){
        W[i] = wi * W[i] / W_E_sum;
    }

    return W;
}

float** exc_act_update(float** exc_act_dummy, float** inh_act_dummy, int bs, int neuron_shape, int leaky,
                       int num_E_nbs, int num_I_nbs, float* W_E, float* W_I, int** N_E, int** N_I){
    float** b = malloc_matrix(bs, neuron_shape);
    for (int k = 0; k < bs; k++){
        float* r = malloc(sizeof(float) * neuron_shape);
        for (int i = 0; i < neuron_shape; i++){
            r[i] = - leaky * exc_act_dummy[k][i];
            for (int j = 0; j < num_E_nbs; j++){
                r[i] += W_E[j] * exc_act_dummy[k][N_E[i][j]];}
            for (int j = 0; j < num_I_nbs; j++){
                r[i] -= W_I[j] * inh_act_dummy[k][N_I[i][j]];}

            b[k][i] = r[i];
        }
        free(r);
    }
    return b;
}

float** inh_act_update(float** exc_act_dummy, float** inh_act_dummy, int bs, int neuron_shape, int leaky,
                       int num_E_nbs, float* W_E, int** N_E){
    float** b = malloc_matrix(bs, neuron_shape);
    for (int k = 0; k < bs; k++){
        float* r = malloc(sizeof(float) * neuron_shape);
        for (int i = 0; i < neuron_shape; i++){
            r[i] = - leaky * inh_act_dummy[k][i];
            for (int j = 0; j < num_E_nbs; j++){
                r[i] += W_E[j] * exc_act_dummy[k][N_E[i][j]];}

            b[k][i] = r[i];
        }
        free(r);
    }
    return b;
}


int main(){

    int neuron_shape = 1600;
    int side_length = sqrt(neuron_shape);
    int bs = 2;
    int re = 3;
    int ri = 5;
    int sigmaE = 3;
    int we = 30;
    int wi = 5;
    int leaky = we + wi;

    int num_E_nbs = get_num_nbs(re);

    int num_I_nbs = get_num_nbs(ri);

//    printf("num_E_nbs = %d\n", num_E_nbs);
//    printf("num_I_nbs = %d\n", num_I_nbs);
//    int r = 3;
//    int* l = get_struc(r);
//    print_int_vector(2*r+1, l);

//    int r = 5;
//    int index = get_index_from_position(7, 5, 8, 8, r);
//    printf("index = %d\n", index);

    int** N_E = compute_indexset(re, num_E_nbs, neuron_shape);
    int** N_I = compute_indexset(ri, num_I_nbs, neuron_shape);

//    print_int_matrix(neuron_shape, num_E_nbs, N_E);
//    printf("\n");
//    print_int_matrix(neuron_shape, num_I_nbs, N_I);
//    printf("\n");

    float* W_E = compute_WE(num_E_nbs, re, we, sigmaE);
    print_float_vector(num_E_nbs, W_E);
    printf("\n");

    float* W_I = compute_WI(num_I_nbs, ri, wi);
    print_float_vector(num_I_nbs, W_I);

}
