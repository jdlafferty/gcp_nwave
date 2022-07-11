#include <stdio.h>
#include <string.h>
#include <stdlib.h>



int main(int argc, char **argv) {
    int test_row = atoi(argv[1]);

    char* param[15];
    char buffer[1024];
    char* record;
    char* line;
    int i = 0, j;

    FILE* fp = fopen("../parameter.csv", "r");
    while((line = fgets(buffer, sizeof(buffer), fp)) != NULL) {
        if (i == test_row) {
            j = 0;
            record = strtok(line, ",");
            while(record != NULL) {
                param[j] = record;
                printf("%s\n", record);
                record = strtok(NULL,",");
                j++;
            }
        }
        i++;
    }
    fclose(fp);

    int ri = atoi(param[0]);
    int re = atoi(param[1]);
    int wi = atoi(param[2]);
    int we = atoi(param[3]);
    int leaky = wi + we;
    int input_dim = 97;
    int neuron_shape = atoi(param[4]);
    int gradient_steps = atoi(param[5]);
    int batch_size = atoi(param[6]);
    float lr_act = atof(param[7]);
    float lr_codebook = atof(param[8]);
    float l0_target = atof(param[9]);
    float threshold = atof(param[10]);

    // TODO: need to add how to write the results back to csv

    return 0;
}