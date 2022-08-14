#include <stdio.h>
#include <math.h>
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
        else {
            int index = core_index + (l[r]-1)/2 + diff + (l[r-(yi - yj)]+1)/2 + (xj - xi);
            free(l);
            return index;
        }
    }
}

int ** compute_indexset(int r, int num_nbs, int neuron_shape){
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

float* compute_W(int num_nbs, int r, int w, int sigmaE){
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

float max(float a, float b){
    if (a >= b){
        return a;
    }
    else{
        return b;
    }
}

void stimulate(int neuron_shape, int bs, float lr_act, float threshold, float eps, float** stimulus,
               float** exc_act_dummy, float** inh_act_dummy, int leaky,
               int num_E_nbs, int num_I_nbs, float* W_E, float* W_I, int** N_E, int** N_I){

    //float relative_error;

    for (int t = 0; t < 50; t++) {

        //float **exc_tm1 = copy_matrix(bs, neuron_shape+1, exc_act_dummy);

        float delta_a_exc, delta_a_inh;

        // Update of activations
        for (int k = 0; k < bs; k++) {
            for (int i = 0; i < neuron_shape; i++) {

                //Update of exhibitory and inhibitory neurons;
                delta_a_exc = - leaky * exc_act_dummy[k][i];
                delta_a_inh = - leaky * inh_act_dummy[k][i];
                for (int j = 0; j < num_E_nbs; j++) {
                    delta_a_exc += W_E[j] * exc_act_dummy[k][N_E[i][j]];
                    delta_a_inh += W_E[j] * exc_act_dummy[k][N_E[i][j]];
                }
                for (int j = 0; j < num_I_nbs; j++) {
                    delta_a_exc -= W_I[j] * inh_act_dummy[k][N_I[i][j]];
                }
                delta_a_exc += stimulus[k][i];
                delta_a_exc = lr_act * delta_a_exc;
                exc_act_dummy[k][i] = exc_act_dummy[k][i] + delta_a_exc;
                exc_act_dummy[k][i] = max(exc_act_dummy[k][i] - threshold, 0.0) - max(-exc_act_dummy[k][i] - threshold, 0.0);

                delta_a_inh = lr_act * delta_a_inh;
                inh_act_dummy[k][i] = inh_act_dummy[k][i] + delta_a_inh;
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

//        if (relative_error < eps) {
//            printf("relative_error = %f\n", relative_error);
//            return exc_act_dummy;
//        } else {
//            printf("relative_error = %f\n", relative_error);
//            printf("Update doesn't converge.");
//            return exc_act_dummy;
//        }

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

int main(int argc, char **argv) {

    int ri = 5;
    int re = 3;
    int wi = 5;
    int we = 30;
    int leaky = wi + we;
    int neuron_shape = 400;
    int sigmaE = 3;
    int bs = 256;
    int imbed_dim = 97;
    float lr_act = 0.01;
    float lr_Phi = 0.01;
    float threshold = 0.01;
    float eps = 5e-3;
    int gradient_step = 10000;
    float l0_target = 0.1;

    float** mat = read_matrix(55529, imbed_dim, "word_embeddings.csv");

    float** Phi = malloc_matrix(imbed_dim, neuron_shape);
    for (int i = 0; i < imbed_dim; i++) {
        for (int j = 0; j < neuron_shape; j++) {
            Phi[i][j] = 0.3 * rand()/(RAND_MAX+1.0);
        }
    }

    //////// These parameters can be precomputed
    int num_E_nbs = get_num_nbs(re);
    int num_I_nbs = get_num_nbs(ri);

    int** N_E = compute_indexset(re, num_E_nbs, neuron_shape);
    int** N_I = compute_indexset(ri, num_I_nbs, neuron_shape);

    float* W_E = compute_W(num_E_nbs, re, we, sigmaE);
    float* W_I = compute_W(num_I_nbs, ri, wi, sigmaE);
    ///////////////// end

    /////// These are all the malloc we need in training
    float** exc_act = malloc_matrix(bs, neuron_shape);

    float** exc_act_dummy = malloc_matrix(bs, neuron_shape + 1);

    float** inh_act_dummy = malloc_matrix(bs, neuron_shape + 1);

    float** stimulus = malloc_matrix(bs, neuron_shape);

    float** fitted_value = malloc_matrix(bs, imbed_dim);

    float** gradient = malloc_matrix(imbed_dim, neuron_shape);

    float** word_batch = malloc_matrix(bs, imbed_dim);
    ///////////////////// end

    for (int g = 0; g < gradient_step; g++){

        // Sample word_batch (bs, imbed_dim) from word_embedding.csv
        sample_matrix(55529, imbed_dim, bs, mat, word_batch);

        for (int i = 0; i < bs; i++) {
            for (int j = 0; j < neuron_shape + 1; j++) {
                exc_act_dummy[i][j] = 0;
            }
        }

        for (int i = 0; i < bs; i++) {
            for (int j = 0; j < neuron_shape + 1; j++) {
                inh_act_dummy[i][j] = 0;
            }
        }

        // stimulus = word_batch @ Phi
        for (int i = 0; i < bs; i++) {
            for (int j = 0; j < neuron_shape; j++) {
                stimulus[i][j] = 0;
            }
        }

        for (int i = 0; i < bs; i++) {
            for (int j = 0; j < neuron_shape; j++) {
                for (int k = 0; k < imbed_dim; k++) {
                    stimulus[i][j] += word_batch[i][k] * Phi[k][j];
                }
            }
        }

        stimulate(neuron_shape, bs, lr_act, threshold, eps, stimulus,
                                  exc_act_dummy, inh_act_dummy, leaky, num_E_nbs, num_I_nbs, W_E, W_I, N_E, N_I);

        // Update of threshold parameter
        float dthreshold = l0_norm(bs, neuron_shape, exc_act_dummy) - l0_target;
        threshold += 0.01 * dthreshold;

        for (int i = 0; i < bs; i++){
            for (int j = 0; j < neuron_shape; j++){
                exc_act[i][j] = exc_act_dummy[i][j];
            }
        }

        //////////////////////// update of codebook

        // fitted_value = exc_act @ Phi.T
        for (int i = 0; i < bs; i++) {
            for (int j = 0; j < imbed_dim; j++) {
                fitted_value[i][j] = 0;
            }
        }

        for (int i = 0; i < bs; i++) {
            for (int j = 0; j < imbed_dim; j++) {
                for (int k = 0; k < neuron_shape; k++) {
                    fitted_value[i][j] += exc_act[i][k] * Phi[j][k];
                }
            }
        }

        // error = word_batch - fitted_value
        for (int i = 0; i < bs; i++) {
            for (int j = 0; j < imbed_dim; j++) {
                fitted_value[i][j] = word_batch[i][j] - fitted_value[i][j];
            }
        }

        float l2_error = l2_loss(bs, imbed_dim, fitted_value);

        // gradient = fitted_value.T @ exc_act
        for (int i = 0; i < imbed_dim; i++) {
            for (int j = 0; j < neuron_shape; j++) {
                gradient[i][j] = 0;
            }
        }

        for (int i = 0; i < imbed_dim; i++) {
            for (int j = 0; j < neuron_shape; j++) {
                for (int k = 0; k < bs; k++) {
                    gradient[i][j] += fitted_value[k][i] * exc_act[k][j];
                }
            }
        }

        normalize(imbed_dim, neuron_shape, gradient);

        scalar_matrix(imbed_dim, neuron_shape, lr_Phi, gradient);

        // Phi += gradient
        for (int i = 0; i < imbed_dim; i++) {
            for (int j = 0; j < neuron_shape; j++) {
                Phi[i][j] += gradient[i][j];
            }
        }

        // normalize Phi
        float result, sum;
        for (int j = 0; j < neuron_shape; j++){
            sum = 0;
            for (int i = 0; i < imbed_dim; i++){
                sum += Phi[i][j] * Phi[i][j];
            }
            if (sqrt(sum) > 1e-8){
                result = sqrt(sum);
            }
            else{
                result = 1e-8;
            }

            for (int i = 0; i < imbed_dim; i++){
                Phi[i][j] = Phi[i][j] / result;
            }
            
        }

        ///////////////////////////////// end

        float l0_loss = l0_norm(bs, neuron_shape, exc_act);
        float l1_loss = l1_norm(bs, neuron_shape, exc_act);

        printf("\n%d. ", g+1);
        printf("l0_loss = %f ", l0_loss);
        printf("l1_loss = %f ", l1_loss);
        printf("l2_loss = %f;  ", l2_error);
        printf("threshold = %f;  \n", threshold);

    }

    write_matrix(imbed_dim, neuron_shape, Phi, "Phi.csv");

    return 0;
}


