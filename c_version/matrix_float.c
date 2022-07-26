#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void print_matrix(int row, int col, float** m) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f ", m[i][j]);
        }
        printf("\n");
    }
}

float** copy_matrix(int row, int col, float** m) {
    float** copy = malloc(sizeof(float*) * row);
    for (int i = 0; i < row; i++) {
        copy[i] = malloc(sizeof(float) * col);
    }

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            copy[i][j] = m[i][j];
        }
    }

    return copy;
}

float** sample_matrix(int row, int col, int sample_size, float** m) {
    float** sample = malloc(sizeof(float*) * sample_size);
    for (int i = 0; i < sample_size; i++) {
        sample[i] = malloc(sizeof(float) * col);
    }

    for (int i = 0; i < sample_size; i++) {
        int m_row = rand() % row;
        for (int j = 0; j < col; j++) {
            sample[i][j] = m[m_row][j];
        }
    }

    return sample;
}

 float** sample_matrix1(int row, int col, int sample_size, float** m) {
     float** sample = malloc(sizeof(float*) * sample_size);
     for (int i = 0; i < sample_size; i++) {
         sample[i] = malloc(sizeof(float) * col);
     }

     for (int i = 0; i < sample_size; i++) {
         for (int j = 0; j < col; j++) {
             sample[i][j] = m[i][j];
         }
     }

     return sample;
 }

float** malloc_matrix(int row, int col) {
    float** result = malloc(sizeof(float*) * row);
    for (int i = 0; i < row; i++) {
        result[i] = malloc(sizeof(float) * col);
    }

    return result;
}

void free_matrix(int row, float ** m) {
    for (int i = 0; i < row; i++) {
        free(m[i]);
    }
    free(m);
}

void print_float_vector(int l, float* v) {
    for (int i = 0; i < l; i++) {
        printf("%f ", v[i]);
    }
    printf("\n");
}

void print_int_vector(int l, int* v){
    for (int i = 0; i < l; i++) {
        printf("%d ", v[i]);
    }
    printf("\n");
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

float** matrix_sum(int r, int c, float** a, float** b) {
    float** p = malloc(sizeof(float*) * r);
    for (int i = 0; i < r; i++) {
        p[i] = malloc(sizeof(float) * c);
    }

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            p[i][j] = 0;
        }
    }

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            p[i][j] = a[i][j] + b[i][j];
        }
    }

    return p;
}

float** matrix_minus(int r, int c, float** a, float** b) {
    float** p = malloc(sizeof(float*) * r);

    for (int i = 0; i < r; i++) {
        p[i] = malloc(sizeof(float) * c);
    }

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            p[i][j] = 0;
        }
    }

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            p[i][j] = a[i][j] - b[i][j];
        }
    }

    return p;
}

void scalar_matrix(int r, int c, float v, float** a) {

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            a[i][j] = v * a[i][j];
        }
    }
}

void square_matrix(int r, int c, float** a) {

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            a[i][j] = a[i][j] * a[i][j];
        }
    }
}

void sqrt_vec(int length, float* v){
    for (int i =0; i<length;i++){
        v[i] = sqrt(v[i]);
    }
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

float sum(int length, float* W) {
    float count = 0;
    for (int i = 0; i < length; i++) {
        count += W[i];
    }
    return count;
}

