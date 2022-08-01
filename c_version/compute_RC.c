#include <stdio.h>
#include <math.h>
#include <string.h>
#include "matrix_float.c"
#include "read_csv.c"

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

float** threshold_func(int bs, int neuron_shape, float threshold, float** exc_act){
    float** m1 = malloc_matrix(bs, neuron_shape);
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < neuron_shape; j++) {
            m1[i][j] = 0;
        }
    }

    float** m2 = malloc_matrix(bs, neuron_shape);
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < neuron_shape; j++) {
            m2[i][j] = 0;
        }
    }

    float** exc_act2 = malloc_matrix(bs, neuron_shape);
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < neuron_shape; j++) {
            exc_act2[i][j] = 0;
        }
    }

    for (int i = 0; i < bs; i++){
        for (int j = 0; j < neuron_shape; j++){
            if (exc_act[i][j] - threshold > 0){
                m1[i][j] = exc_act[i][j] - threshold;
            }
            else{
                m1[i][j] = 0;
            }
        }
    }

    for (int i = 0; i < bs; i++){
        for (int j = 0; j < neuron_shape; j++){
            if (- exc_act[i][j] - threshold > 0){
                m2[i][j] = - exc_act[i][j] - threshold;
            }
            else{
                m2[i][j] = 0;
            }
        }
    }

    for (int i = 0; i < bs; i++){
        for (int j = 0; j < neuron_shape; j++){
            exc_act2[i][j] = m1[i][j] - m2[i][j];
        }
    }

    free_matrix(bs, m1);

    free_matrix(bs, m2);

    return exc_act2;
}

float** stimulate(int neuron_shape, int bs, float lr_act, float threshold, float eps, float** stimulus, float** exc_act
        , float** inh_act, float** exc_act_dummy, float** inh_act_dummy, int leaky,
                  int num_E_nbs, int num_I_nbs, float* W_E, float* W_I, int** N_E, int** N_I){

    float relative_error;

    for (int t = 0; t < 50; t++){

        float** exc_tm1 = copy_matrix(bs, neuron_shape, exc_act);

        float** delta_a_exc = exc_act_update(exc_act_dummy, inh_act_dummy, bs, neuron_shape, leaky,
                                             num_E_nbs, num_I_nbs, W_E, W_I, N_E, N_I);

        float** delta_a_inh = inh_act_update(exc_act_dummy, inh_act_dummy, bs, neuron_shape, leaky,
                                             num_E_nbs, W_E, N_E);

        // Compute of exc_act;
        float** stimulus_delta_exc = matrix_sum(bs, neuron_shape, stimulus, delta_a_exc);
        free_matrix(bs, delta_a_exc);

        scalar_matrix(bs, neuron_shape, lr_act, stimulus_delta_exc);

        float** exc_act_new = matrix_sum(bs, neuron_shape, exc_act, stimulus_delta_exc);
        free_matrix(bs, exc_act);
        free_matrix(bs, stimulus_delta_exc);
        exc_act = exc_act_new;

        //compute of inh_act;
        scalar_matrix(bs, neuron_shape, lr_act, delta_a_inh);

        float** inh_act_new = matrix_sum(bs, neuron_shape, inh_act, delta_a_inh);
        free_matrix(bs, inh_act);
        free_matrix(bs, delta_a_inh);
        inh_act = inh_act_new;

        /*printf("exc_act = ");
        print_matrix(bs, neuron_shape, exc_act);
        printf("\n");*/

        exc_act_new = threshold_func(bs, neuron_shape, threshold, exc_act);
        free_matrix(bs, exc_act);
        exc_act = exc_act_new;

        inh_act_new = threshold_func(bs, neuron_shape, threshold, inh_act);
        free_matrix(bs, inh_act);
        inh_act = inh_act_new;

        for (int i = 0; i < bs; i++){
            for (int j = 0; j < neuron_shape; j++){
                exc_act_dummy[i][j] = exc_act[i][j];
                inh_act_dummy[i][j] = inh_act[i][j];
            }
        }

        //printf("%d exc_act[1][1] = %f", t, exc_act[1][1]);
        //print_matrix(bs, neuron_shape, exc_act);
        //printf("\n");

        float** da = matrix_minus(bs, neuron_shape, exc_act, exc_tm1);

        float sqrt_da = 0;
        for (int i = 0; i < bs; i++){
            for (int j = 0; j < neuron_shape; j++){
                sqrt_da += da[i][j] * da[i][j];
            }
        }
        sqrt_da = sqrt(sqrt_da);

        float sqrt_exc_tm1 = 0;
        for (int i = 0; i < bs; i++){
            for (int j = 0; j < neuron_shape; j++){
                sqrt_exc_tm1 += exc_tm1[i][j] * exc_tm1[i][j];
            }
        }
        sqrt_exc_tm1 = sqrt(sqrt_exc_tm1);

        relative_error = sqrt_da / (eps + sqrt_exc_tm1);

        free_matrix(bs, da);

        free_matrix(bs, exc_tm1);
    }

    if (relative_error < eps){
        printf("relative_error = %f\n", relative_error);
        return exc_act;
    }
    else{
        printf("relative_error = %f ", relative_error);
        printf("Update doesn't converge.");
        return exc_act;
    }

}

void normalize(int row, int col, float** w){
    for (int i =0; i < row; i++){
        float mean = sum(col, w[i])/col;
        for (int j = 0; j < col; j++){
            w[i][j] = w[i][j] - mean;
        }
    }
}

float absolute(float a){
    if (a > 0){
        return a;
    }
    else if (a == 0){
        return 0;
    }
    else{
        return -1 * a;
    }
}

float* Phi_normalize(int row, int col, float** w){
    float* result = malloc(sizeof(float) * col);
    for (int i = 0; i < col; i++){
        float sum = 0;
        for (int j = 0; j < row; j++){
            sum += w[j][i] * w[j][i];
        }
        if (sqrt(sum) > 1e-8){
            result[i] = sqrt(sum);
        }
        else{
            result[i] = 1e-8;
        }
    }
    return result;
}

float l0_norm(int r, int c, float** w){
    float count = 0;
    int size = r * c;
    for (int i = 0; i<r; i++){
        for (int j = 0; j < c; j++){
            if (absolute(w[i][j]) > 1e-4){
                count+=1;
            }
        }
    }
    return count/size;
}

float l1_norm(int r, int c, float** w){
    float sum = 0;
    int size = r*c;
    for (int i = 0; i < r; i++){
        for (int j  = 0; j< c; j++){
            sum += absolute(w[i][j]);
        }
    }

    sum = sum/(size);
    return sum;
}

float l2_loss(int r, int c, float** w){
    float loss = 0;
    for (int i = 0; i < r; i++){
        for (int j = 0; j< c; j++){
            loss += w[i][j] * w[i][j];
        }
    }
    float sqrtloss = sqrt(loss);
    return sqrtloss;
}

float** update_Phi(float** word_batch, float** exc_act, int bs, int imbed_dim, int neuron_shape, float lr_Phi, float** Phi, float l2_error){
    float** Phi_T = transpose(imbed_dim, neuron_shape, Phi);

    float** fitted_value = multiply(bs, neuron_shape, imbed_dim, exc_act, Phi_T);
    free_matrix(neuron_shape, Phi_T);

    float** error = matrix_minus(bs, imbed_dim, word_batch, fitted_value);
    l2_error = l2_loss(bs, imbed_dim, error);
    free_matrix(bs, fitted_value);

    float** error_T = transpose(bs, imbed_dim, error);
    free_matrix(bs, error);

    float** gradient = multiply(imbed_dim, bs, neuron_shape, error_T, exc_act);
    free_matrix(imbed_dim, error_T);

    normalize(imbed_dim, neuron_shape, gradient);

    scalar_matrix(imbed_dim, neuron_shape, lr_Phi, gradient);

    float** Phi_new = matrix_sum(imbed_dim, neuron_shape, Phi, gradient);
    free_matrix(imbed_dim, gradient);
    //free_matrix(imbed_dim, Phi);
    Phi = Phi_new;

    float* normalize = Phi_normalize(imbed_dim, neuron_shape, Phi);

    for (int j = 0; j < neuron_shape; j++){
        for (int i = 0; i < imbed_dim; i++){
            Phi[i][j] = Phi[i][j] / normalize[j];
        }
    }

    free(normalize);
    return Phi;
}



int main() {

    int ri = 5;
    int re = 3;
    int wi = 5;
    int we = 30;
    int leaky = wi + we;
    int neuron_shape = 400;
    int sigmaE = 3;
    int imbed_dim = 97;
    float lr_act = 0.01;
    float threshold = 0.01;
    float eps = 5e-3;

    int num_E_nbs = get_num_nbs(re);
    int num_I_nbs = get_num_nbs(ri);

    int** N_E = compute_indexset(re, num_E_nbs, neuron_shape);
    int** N_I = compute_indexset(ri, num_I_nbs, neuron_shape);

    float* W_E = compute_W(num_E_nbs, re, we, sigmaE);
    float* W_I = compute_W(num_I_nbs, ri, wi, sigmaE);

    float** codebook = read_matrix(imbed_dim, neuron_shape, "Phi.csv");
    //print_matrix(imbed_dim, neuron_shape, codebook);

    float** batch = malloc_matrix(imbed_dim, imbed_dim);
    for (int i = 0; i < imbed_dim; i++){
        for (int j  = 0; j < imbed_dim; j++){
            batch[i][i] = 1.0;
        }
    }

    for (int i = 0; i < imbed_dim; i++){
        for (int j  = 0; j < imbed_dim; j++){
            batch[i][j] -= 1.0/imbed_dim;
        }
    }

    float std_value = std(imbed_dim, batch[1]);

    for (int i = 0; i < imbed_dim; i++){
        for (int j  = 0; j < imbed_dim; j++){
            batch[i][j] /= std_value;
        }
    }

    float** stimulus1 = multiply(imbed_dim, imbed_dim, neuron_shape, batch, codebook);

//    printf("stimulus1 = ");
//    print_matrix(imbed_dim, neuron_shape, stimulus1);

    float** exc_act1 = malloc_matrix(imbed_dim, neuron_shape);
    for (int i = 0; i < imbed_dim; i++) {
        for (int j = 0; j < neuron_shape; j++) {
            exc_act1[i][j] = 0;
        }
    }

    float** inh_act1 = malloc_matrix(imbed_dim, neuron_shape);
    for (int i = 0; i < imbed_dim; i++) {
        for (int j = 0; j < neuron_shape; j++) {
            inh_act1[i][j] = 0;
        }
    }

    float** exc_act_dummy1 = malloc_matrix(imbed_dim, neuron_shape + 1);
    for (int i = 0; i < imbed_dim; i++) {
        for (int j = 0; j < neuron_shape + 1; j++) {
            exc_act_dummy1[i][j] = 0;
        }
    }

    float** inh_act_dummy1 = malloc_matrix(imbed_dim, neuron_shape + 1);
    for (int i = 0; i < imbed_dim; i++) {
        for (int j = 0; j < neuron_shape + 1; j++) {
            inh_act_dummy1[i][j] = 0;
        }
    }

    printf("\n RC part: threshold = %f; ", threshold);
    float** RC = stimulate(neuron_shape, imbed_dim, lr_act, threshold, eps, stimulus1, exc_act1,
                           inh_act1, exc_act_dummy1, inh_act_dummy1,
                           leaky, num_E_nbs, num_I_nbs, W_E, W_I, N_E, N_I);

    write_matrix(imbed_dim, neuron_shape, RC, "RC.csv");


    return 0;
}


