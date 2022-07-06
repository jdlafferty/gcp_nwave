#include <stdio.h>
#include <math.h>

void print_matrix(int row, int col, int m[row][col]) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%d ", m[i][j]);
        }
        printf("\n");
    }
}

void transpose(int row, int col, int a[row][col], int b[col][row]) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            b[j][i] = a[i][j];
        }
    }
}

void multiply(int r1, int c1, int c2, int a[r1][c1], int b[c1][c2], int c[r1][c2]) {
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
}

int dot(int l, int a[l], int b[l]) {
    int c = 0;
    for (int i = 0; i < l; i++) {
        c += a[i] * b[i];
    }
    return c;
}

float norm(int l, int v[l]) {
    return sqrt(dot(l, v, v));
}

void right_multiply(int row, int col, int m[row][col], int v[col], int c[row]) {
    for (int i = 0; i < row; i++) {
        c[i] = 0;
    }

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            c[i] += m[i][j] * v[j];
        }
    }
}

void left_multiply(int row, int col, int v[row], int m[row][col], int c[col]) {
    for (int j = 0; j < col; j++) {
        c[j] = 0;
    }

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            c[j] += v[i] * m[i][j];
        }
    }
}





int main() {
    int m[3][3] = {
    10, 23, 42,    
    1, 654, 0,  
    40652, 22, 0  
    };  
    print_matrix(3, 3, m);
    return 0;
}
