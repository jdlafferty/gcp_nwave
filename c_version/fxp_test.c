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

int main(){
    float leaky = 35.0;
    float test_variable = -0.003290;
    float test_variable2 = -0.002287;

    printf("fixed_point_leaky = %d\n", convert_to_fix_point(leaky));
    printf("fixed_point_variable = %d\n", convert_to_fix_point(test_variable));
    printf("fixed_point_variable2 = %d\n\n", convert_to_fix_point(test_variable2));

    // leaky * test_variable
    float test_result = leaky * test_variable;
    printf("test_result: 35 * -0.003290 = %f\n", test_result);
    printf("fixedpoint_test_result: 35 * -0.003290 = %d\n\n", convert_to_fix_point(test_result));

    // test_variable + test_variable2
    float test_result2 = test_variable + test_variable2;
    printf("test_result2: -0.003290 + -0.002287 = %f\n", test_result2);
    printf("fixedpoint_test_result2: -0.003290 + -0.002287 = %d\n\n", convert_to_fix_point(test_result2));

    // test_variable - test_variable3
    float test_result3 = test_variable - test_variable2;
    printf("test_result3: -0.003290 - -0.002287 = %f\n", test_result3);
    printf("fixedpoint_test_result3: -0.003290 - -0.002287 = %d\n", convert_to_fix_point(test_result3));

}
