CC = gcc
CFLAGS = -std=c99 -Wall -pedantic -g3

train_msg_passing: train_msg_passing.c
	${CC} -o $@ ${CFLAGS} $^

compute_RC: compute_RC.c
	${CC} -o $@ ${CFLAGS} $^

all: train_msg_passing compute_RC
