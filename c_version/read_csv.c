#include <stdio.h>
#include <string.h>
#include <stdlib.h>

float** read_matrix(int row, int col, char* filename) {
    float** mat = malloc(sizeof(float*) * row);
    for (int i = 0; i < row; i++) {
        mat[i] = malloc(sizeof(float) * col);
    }

    char buffer[1024];
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

int main(){
    float** mat = read_matrix(55529, 97, "word_embeddings.csv");
    write_matrix(55529, 97, mat, "word_embeddings.csv");
    return 0;
}
