cmake_minimum_required(VERSION 3.22)
project(c_version C)

set(CMAKE_C_STANDARD 11)

add_executable(c_version
        compute_act_laplacian.c
        compute_act_msg_passing.c
        compute_RC.c
        matrix_float.c
        msg_passing_algorithm.c
        read_csv.c
        read_param.c
        test.c
        train_laplacian.c
        train_msg_passing.c)
