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
    //int count = 0;
    float** mat = read_matrix(1, 1600, "stimulus.csv");
//    for (int i = 0; i < 1; i++){
//        for (int j = 0; j < 1600; j++){
//            if (mat[i][j] > 1.024 || mat[i][j] < -1.024){
//                mat[i][j] = 0.01 * mat[i][j];
//                count+=1;
//                int x = j/40;
//                int y = j - 40 * x;
//                printf("a[%d][%d] = %d\n", x, y, convert_to_fix_point(mat[i][j]));
//                //printf("mat[%d][%d] = %f\n", i, j, mat[i][j]);
//            }
//        }
//    }

    //printf("count = %d", count);

    for (int i = 0; i < 1; i++){
        for (int j = 0; j < 1600; j++){
            int x = j/40;
            int y = j - 40 * x;
            if (x == 29 && y == 0){
            printf("a[%d][%d] = %d\n", x, y, convert_to_fix_point(mat[i][j]));}
        }}

    for (int i = 0; i < 1; i++){
        for (int j = 0; j < 1600; j++){
            int x = j/40;
            int y = j - 40 * x;
            if (x == 2 && y == 35){
            printf("a1[%d][%d] = %d\n", x, y, convert_to_fix_point(0.01024 * mat[i][j]));}
        }}

//    printf("%f\n", mat[0][95]);  [2][15]
//    printf("%f\n", mat[0][115]);   [2][35]

}