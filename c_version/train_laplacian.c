#include <stdio.h>
#include <math.h>
#include <string.h>
#include "matrix_float.c"
#include "read_csv.c"

int find_distsq(int i, int j, int side_length){
    int xi = i/side_length;
    int yi = i % side_length;
    int xj = j/side_length;
    int yj = j%side_length;
    int dist = (xi-xj)*(xi-xj) + (yi-yj)*(yi-yj);
    return dist;
}

float** get_laplacian_matrix(int n, int r1, int r2, int wi, int we, int sigmaE){
    float** laplacian = malloc_matrix(n, n);

    int r1sq = r1*r1;
    int r2sq = r2*r2;
    int side_length = sqrt(n);

    float** We = malloc_matrix(n, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            We[i][j] = 0;
        }
    }

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            int distsq = find_distsq(i, j, side_length);
            /*printf("distsq = %d\n", distsq);*/
            if (distsq <= r1sq){
                /*printf("r1sq = %d\n", r1sq);*/
                We[i][j] = - we * exp(-distsq/2.0/sigmaE);
                /*printf("We[%d][%d] = %f\n", i, j, We[i][j]);*/
            }
            else{
                We[i][j] = 0;
            }
        }
    }

    for (int i = 0; i < n; i++){
        float  s = - sum(n, We[i]);
        for (int j = 0; j < n; j++){
            We[i][j] = we * We[i][j] / s;
        }
    }

    float** Wi = malloc_matrix(n, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Wi[i][j] = 0;
        }
    }

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            int distsq = find_distsq(i, j, side_length);
            if (distsq <= r2sq){
                Wi[i][j] = wi;
            }
            else{
                Wi[i][j] = 0;
            }
        }
    }

    for (int i = 0; i < n; i++){
        float  s = sum(n, Wi[i]);
        for (int j = 0; j < n; j++){
            Wi[i][j] = wi * Wi[i][j] / s;
        }
    }

    laplacian = matrix_sum(n, n, We, Wi);

    for (int i = 0; i < n; i++){
        laplacian[i][i] = wi + we;
    }

    free_matrix(n, We);

    free_matrix(n, Wi);

    return laplacian;
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
        , float** laplacian){

    float relative_error;

    for (int t = 0; t < 50; t++){

        float** exc_tm1 = copy_matrix(bs, neuron_shape, exc_act);


        float** exc_laplacian = multiply(bs, neuron_shape, neuron_shape, exc_act, laplacian);

        /*printf("exc_act_laplacian = ");
        print_matrix(bs, neuron_shape, exc_act_laplacian);
        printf("\n");*/

        float** stimulus_laplacian = matrix_minus(bs, neuron_shape, stimulus, exc_laplacian);
        free_matrix(bs, exc_laplacian);

        /*printf("stimulus_laplacian = ");
        print_matrix(bs, neuron_shape, stimulus_laplacian);
        printf("\n");*/

        scalar_matrix(bs, neuron_shape, lr_act, stimulus_laplacian);

        /*printf("stimulus_laplacian1 = ");
        print_matrix(bs, neuron_shape, stimulus_laplacian);
        printf("\n");*/


        float** exc_act_new = matrix_sum(bs, neuron_shape, exc_act, stimulus_laplacian);
        free_matrix(bs, exc_act);


        free_matrix(bs, stimulus_laplacian);

        exc_act = exc_act_new;

        /*printf("exc_act = ");
        print_matrix(bs, neuron_shape, exc_act);
        printf("\n");*/

        exc_act_new = threshold_func(bs, neuron_shape, threshold, exc_act);
        free_matrix(bs, exc_act);
        exc_act = exc_act_new;

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
        printf("relative_error = %f", relative_error);
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



int main(int argc, char **argv) {

    int ri = 5;
    int re = 3;
    int wi = 5;
    int we = 30;
    int neuron_shape = 100;
    int sigmaE = 3;
    int bs = 256;
    int imbed_dim = 97;
    float lr_act = 0.01;
    float lr_Phi = 0.01;
    float threshold = 0.007;
    float eps = 5e-3;
    int gradient_step = 50000;
    float l0_target = 0.1;

    float** mat = read_matrix(55529, imbed_dim, "word_embeddings.csv");

//    printf("Word_batch = ");
//    print_matrix(bs, imbed_dim, word_batch);
//    printf("\n");

    float** laplacian = get_laplacian_matrix(neuron_shape, re, ri, wi, we, sigmaE);
    //print_matrix(neuron_shape, neuron_shape, laplacian);

    //float** Phi = read_matrix(97, 1600, "code_book.csv");
    //print_matrix(97, 1600, Phi);

    float** Phi = malloc_matrix(imbed_dim, neuron_shape);
    for (int i = 0; i < imbed_dim; i++) {
        for (int j = 0; j < neuron_shape; j++) {
            Phi[i][j] = rand()/(RAND_MAX+1.0);
        }
    }

    for (int i = 0; i < gradient_step; i++){

        float** word_batch = sample_matrix(55529, imbed_dim, bs, mat);
//        printf("word_batch = ");
//        print_matrix(bs, imbed_dim, word_batch);
//        printf("\n");

        float** stimulus = multiply(bs, imbed_dim, neuron_shape, word_batch, Phi);

        float** exc_act = malloc_matrix(bs, neuron_shape);
        for (int i = 0; i < bs; i++) {
            for (int j = 0; j < neuron_shape; j++) {
                exc_act[i][j] = 0;
            }
        }

        exc_act = stimulate(neuron_shape, bs, lr_act, threshold, eps, stimulus, exc_act, laplacian);

        float dthreshold = l0_norm(bs, neuron_shape, exc_act) - l0_target;
        threshold += 0.01 * dthreshold;

        //////////////////////// update of codebook

        float** Phi_T = transpose(imbed_dim, neuron_shape, Phi);

        float** fitted_value = multiply(bs, neuron_shape, imbed_dim, exc_act, Phi_T);
        free_matrix(neuron_shape, Phi_T);

        float** error = matrix_minus(bs, imbed_dim, word_batch, fitted_value);
        float l2_error = l2_loss(bs, imbed_dim, error);
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

        ///////////////////////////////// end

        float l0_loss = l0_norm(bs, neuron_shape, exc_act);
        float l1_loss = l1_norm(bs, neuron_shape, exc_act);

        printf("\n%d. ", i+1);
        printf("l0_loss = %f ", l0_loss);
        //printf("l1_loss = %f\n", l1_loss);
        printf("l2_loss = %f;  ", l2_error);
        printf("threshold = %f;  \n", threshold);

        //printf("Phi[1][1] = %f\n", Phi[1][1]);
        //print_matrix(imbed_dim, neuron_shape, Phi);

        free_matrix(bs, exc_act);

    }

    printf("\n");
    print_matrix(imbed_dim, neuron_shape, Phi);

    write_matrix(imbed_dim, neuron_shape, Phi, "Phi.csv");

    return 0;
}

