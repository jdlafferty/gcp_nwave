#include <stdio.h>
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


int main(int argc, char **argv) {
    int test_row = atoi(argv[1]);

    char* param[15];
    char buffer[1024];
    char* record;
    char* line;
    int i = 0, j;

    FILE* fp = fopen("../parameter.csv", "r");
    while((line = fgets(buffer, sizeof(buffer), fp)) != NULL) {
        if (i == test_row) {
            j = 0;
            record = strtok(line, ",");
            while(record != NULL) {
                param[j] = record;
                //printf("%s\n", record);
                record = strtok(NULL,",");
                j++;
            }
        }
        i++;
    }
    fclose(fp);

    int ri = atoi(param[0]);
    int re = atoi(param[1]);
    int wi = atoi(param[2]);
    int we = atoi(param[3]);
    int leaky = wi + we;
    int imbed_dim = 97;
    int neuron_shape = atoi(param[4]);
    int gradient_steps = atoi(param[5]);
    int bs = atoi(param[6]);
    float lr_act = atof(param[7]);
    float lr_codebook = atof(param[8]);
    float l0_target = atof(param[9]);
    float threshold = atof(param[10]);
    int sigmaE = 3;
    float eps = 5e-3;

    printf("ri = %d\n", ri);
    printf("re = %d\n", re);
    printf("wi = %d\n", wi);
    printf("we = %d\n", we);
    printf("imbed_dim = %d\n", imbed_dim);
    printf("neuron_shape = %d\n", neuron_shape);
    printf("bs = %d\n", bs);
    printf("lr_act = %f\n", lr_act);
    printf("threshold = %f\n", threshold);

//    int ri = 5;
//    int re = 3;
//    int wi = 5;
//    int we = 30;
//    int neuron_shape = 1600;
//    int sigmaE = 3;
//    int bs = 256;
//    int imbed_dim = 97;
//    float lr_act = 0.01;
//    float threshold = 0.005;
//    float eps = 5e-3;

    // TODO: need to add how to write the results back to csv

    float** mat = read_matrix(55529, imbed_dim, "word_embeddings.csv");
    float** word_batch = sample_matrix(55529, imbed_dim, bs, mat);

//    printf("Word_batch = ");
//    print_matrix(bs, imbed_dim, word_batch);
//    printf("\n");

    float** Phi = malloc_matrix(imbed_dim, neuron_shape);
    for (int i = 0; i < imbed_dim; i++) {
        for (int j = 0; j < neuron_shape; j++) {
            Phi[i][j] = 0.5;
        }
    }

    float** laplacian = get_laplacian_matrix(neuron_shape, re, ri, wi, we, sigmaE);

    float** stimulus = multiply(bs, imbed_dim, neuron_shape, word_batch, Phi);

    float** exc_act = malloc_matrix(bs, neuron_shape);
    for (int i = 0; i < bs; i++) {
        for (int j = 0; j < neuron_shape; j++) {
            exc_act[i][j] = 0;
        }
    }

    exc_act = stimulate(neuron_shape, bs, lr_act, threshold, eps, stimulus, exc_act, laplacian);

    return 0;
}

