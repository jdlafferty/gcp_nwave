#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "matrix_float.c"

float** read_matrix(int row, int col, char* filename) {
    float** mat = malloc(sizeof(float*) * row);
    for (int i = 0; i < row; i++) {
        mat[i] = malloc(sizeof(float) * col);
    }

    char buffer[col * 10];
    char* record;
    char* line;
    int i = 0, j;

    FILE* fp = fopen(filename, "r");
    while((line = fgets(buffer, sizeof(buffer), fp)) != NULL) {
        j = 0;
        record = strtok(line, ",");
        while(record != NULL) {
            mat[i][j] = atof(record);
            record = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    fclose(fp);

    return mat;
}

void write_matrix(int row, int col, float** mat, char* filename) {
    FILE *fp = fopen(filename, "w"); 
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            fprintf(fp, "%f", mat[i][j]);
            if (j != col - 1) {fprintf(fp, ",");}
            else {fprintf(fp, "\n");}
        }
    }
    fclose(fp);
}



 int main() {
     float** mat = read_matrix(256, 1600, "stimulus.csv");
     print_matrix(256, 1600, mat);
//     printf("mat[0][166] = %f", mat[0][0]);
     // write_matrix(55529, 97, mat, "word_embeddings.csv");


     //get_word(4);

     for (int i = 0; i < 97; i++) {
          free(mat[i]);
      }
     free(mat);

     return 0;
 }
