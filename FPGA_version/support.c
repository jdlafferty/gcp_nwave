#include <act/actsim_ext.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int num_weights = 0;
static double *weights = NULL;

#define FIX_POINT_A 7
#define FIX_POINT_B 17

float** sample_matrix(int row, int col, int sample_size, float** m) {
    float** sample = malloc(sizeof(float*) * sample_size);
    for (int i = 0; i < sample_size; i++) {
         sample[i] = malloc(sizeof(float) * col);
     }

    for (int i = 0; i < sample_size; i++) {
        int m_row = rand() % row;
        printf("row = %d\n", m_row);
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

void print_matrix(int row, int col, float** m) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f ", m[i][j]);
        }
        printf("\n");
    }
}


int set_num_weights (int argc, long *args)
{
  int i;
    
  if (argc != 1) {
    printf ("Error: set_num_weights takes one argument!\n");
    return 0;
  }
  if (weights) {
	free (weights);
  }
  num_weights = args[0];
  weights = (double *) malloc (sizeof (double) * num_weights);
  if (!weights) { 
    printf ("Error: could not allocate weights!\n");
    num_weights = 0;
    return 0;
  }
  for (i=0; i < num_weights; i++) { 
    weights[i] = 0;
  }
  return 1;
}

#define sigmaE 3.0
#define we 30.0
#define wi 5.0

/*
 * arg0 = weight number
 * arg1, arg2 = dx, dy
 * arg3 = 0 => inhibitory, 1 = excitatory
 *
 */
int set_weight (int argc, long *args)
{
  long dx, dy;
  
  if (argc != 4) {
    printf ("Error: set_weight needs 4 arguments\n");
    return 0;
  }

  dx = args[1];
  dy = args[2];

  if (args[3] == 0) {
    weights[args[0]] = 1;
  }
  else {
    weights[args[0]] = exp(-(args[1]*args[1] + args[2]*args[2])/2.0/sigmaE);
  }  
  return 1;
}

int get_exc_weight (int argc, long *args)
{
  double total;
  int i;
  
  if (argc != 1) {
    printf ("Error: get_weight needs 1 argument\n");
    return 0;
  }
  if (args[0] < 0 || args[0] >= num_weights) {
    printf ("Error: get_weight (%lu) out of range (%d)\n", args[0], num_weights);
    return 0;
  }    

  total = 0;
  for (i=0; i < num_weights; i++) {
    total += weights[i];
  }
  i = ((we*weights[args[0]]/total)*(1 << FIX_POINT_B)+0.5);  // Should multiple a constant: we
  return i;
}

int get_inh_weight (int argc, long *args)
{
  double total;
  int i;
  
  if (argc != 1) {
    printf ("Error: get_weight needs 1 argument\n");
    return 0;
  }
  if (args[0] < 0 || args[0] >= num_weights) {
    printf ("Error: get_weight (%lu) out of range (%d)\n", args[0], num_weights);
    return 0;
  }    

  total = 0;
  for (i=0; i < num_weights; i++) {
    total += weights[i];
  }
  i = (wi*(weights[args[0]]/total)*(1 << FIX_POINT_B)+0.5);  // Should multiple a constant: wi
  return i;
}


// Read .csv file as a matrix
float** read_matrix(int row, int col, char* filename) {
    float** mat = malloc(sizeof(float*) * row);
    for (int i = 0; i < row; i++) {
        mat[i] = malloc(sizeof(float) * col);
    }

    char buffer[col*10];
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

float **_internal_stimulus = NULL;
int _shape = -1;

int init_stim (int argc, long *args)
{
  float f;
  int tmp;

  if (argc != 2) {
    printf ("Error: init_stim needs two arguments\n");
    return 0;
  }
  // args[0] = batch_size
  // args[1] = neuron_shape

  // f should be set to the stim for neuron at (x,y)
  // where x = args[0], y = args[1]
 
  _shape = sqrt(args[1]);
  //printf("neuron_shape = %d ", shape);

  // float** mat = read_matrix(55529, 97, "word_embeddings.csv");
  // float** word_batch = sample_matrix1(55529, 97, args[2], mat);
  // float** Phi = read_matrix(97, args[3], "codebook.csv");
  // float** stimulus = multiply(args[2], 97, args[3], word_batch, Phi); 
  if (_internal_stimulus) {
    free (_internal_stimulus);
  }

  float** mat = read_matrix(55529, 97, "word_embeddings.csv");
  float** word_batch = sample_matrix(55529, 97, args[0], mat);
  float** Phi = read_matrix(97, args[1], "codebook.csv");
  _internal_stimulus = multiply(args[0], 97, args[1], word_batch, Phi); 
  //_internal_stimulus = read_matrix(args[0], args[1], "stimulus_row2.csv");

  return 0;
}

struct expr_res get_stim (int argc, struct expr_res *args)
{
  struct expr_res ret;
  float f;
  int tmp;

  ret.width = FIX_POINT_A + FIX_POINT_B;

  if (argc != 3) {
    printf ("Error: get_stim needs three arguments\n");
    return ret;
  }
  // args[0] = ith row in the batch_size
  // args[1] = x coordinate
  // args[2] = y coordinate

  // f should be set to the stim for neuron at (x,y)
  // where x = args[1], y = args[2]
 
  //printf("neuron_shape = %d ", shape);

  // float** mat = read_matrix(55529, 97, "word_embeddings.csv");
  // float** word_batch = sample_matrix1(55529, 97, args[2], mat);
  // float** Phi = read_matrix(97, args[3], "codebook.csv");
  // float** stimulus = multiply(args[2], 97, args[3], word_batch, Phi); 

  int l = args[1].v * _shape  + args[2].v;
  //printf ("x = %ld, and y = %ld; \n", args[0].v, args[1].v);
  f = _internal_stimulus[args[0].v][l];  
  //printf ("stimulus[%ld][%d] = %f\n", args[0].v, l, f);

  if (f >= 0) {
     tmp = (int) (f * (1 << FIX_POINT_B) + 0.5);
  } 
  else {
     tmp = (int) ((-f) * (1 << FIX_POINT_B) + 0.5);
     tmp = (1 << (FIX_POINT_A + FIX_POINT_B)) - tmp;
  }
  ret.v = tmp;
  return ret;
}

// int main(){
//   // long parameter[4];
//   // parameter[0] = 2;
//   // parameter[1] = 9;
//   // parameter[2] = 1;
//   // parameter[3] = 1600;

//   // get_stim(4, parameter);

//   long Le = 12;
//   long i = 
//   set_num_weights(1, Le);

// }
