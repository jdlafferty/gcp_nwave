#include <stdio.h>
#include <math.h>
#include <string.h>
#include "matrix_float.c"
#include "read_csv.c"

float max(float a, float b){
    if (a >= b){
        return a
    }
    else{
        return b
    }
}

void threshold_func(int bs, int neuron_shape, float threshold, float** act){
    for (int i = 0; i < bs; i++){
        for (int j = 0; j < neuron_shape; j++){
            exc_act[i][j] = max(act - threshold, 0.0) - max(-act - threshold, 0.0)
        }
    }
}

int main(){

    

}
