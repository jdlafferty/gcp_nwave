#include <stdio.h>
#include <math.h>

void print_matrix(int row, int col, float** m) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f ", m[i][j]);
        }
        printf("\n");
    }
}

float** transpose(int row, int col, float** a) {
    float** b = malloc(sizeof(float*) * col);
     
    for (int j = 0; j < col; j++) {
        b[j] = malloc(sizeof(float) * row);
    }

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            b[j][i] = a[i][j];
        }
    }

    return b;
}

float** multiply(int r1, int c1, int c2, float** a, float** b) {
    float** c = malloc(sizeof(float*) * r1);
     
    for (int i = 0; i < r1; i++) {
        c[i] = malloc(sizeof(float) * c2);
    }

    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            c[i][j] = 0;
        }
    }

    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            for (int k = 0; k < c1; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return c;
}

float dot(int l, float* a, float* b) {
    float c = 0.0;
    for (int i = 0; i < l; i++) {
        c += a[i] * b[i];
    }
    return c;
}

float norm(int l, float* v) {
    return sqrt(dot(l, v, v));
}

float* right_multiply(int row, int col, float** m, float* v) {
    float* c = malloc(sizeof(float) * row);

    for (int i = 0; i < row; i++) {
        c[i] = 0;
    }

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            c[i] += m[i][j] * v[j];
        }
    }

    return c;
}

float* left_multiply(int row, int col, float* v, float** m) {
    float* c = malloc(sizeof(float) * col);

    for (int j = 0; j < col; j++) {
        c[j] = 0;
    }

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            c[j] += v[i] * m[i][j];
        }
    }

    return c;
}
