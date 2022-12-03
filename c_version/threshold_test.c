#include <stdio.h>
#include <math.h>
#include <string.h>
#include "matrix_float.c"
#include "read_csv.c"

#define FIX_POINT_A 4
#define FIX_POINT_B 10

int convert_to_fix_point (float f){
    int tmp;

    if (f >= 0) {
        tmp = (int) (f * (1 << FIX_POINT_B) + 0.5);
    }
    else {
        tmp = (int) ((-f) * (1 << FIX_POINT_B) + 0.5);
        tmp = (1 << (FIX_POINT_A + FIX_POINT_B)) - tmp;
    }
    return tmp;
}

float max(float a, float b){
    if (a >= b){
        return a;
    }
    else{
        return b;
    }
}

float softmax(float f , float threshold){
        return max(f - threshold, 0.0) - max(-f - threshold, 0.0);
}

int main(){
    float f = -1.047523; // The value of softmax function.
    float threshold = 0.01024;

    printf("x = %d", convert_to_fix_point(-0.003290));

    float result = softmax(f, threshold);
    printf("result = %f\n", result);
    int result_fixed = convert_to_fix_point(result);  // Result in fixed point number.
    printf("result_fixed = %d\n", result_fixed);

    int f_fixed = convert_to_fix_point(f);  // The value of softmax function in fixed point number.
    int threshold_fixed = convert_to_fix_point(threshold); //threshold in fixed point number.

    printf("f_fixed = %d\n", f_fixed);
    printf("threshold_fixed = %d\n", threshold_fixed);


}

