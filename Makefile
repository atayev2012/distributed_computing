CC = mpicc
TARGET_TASK_1 = ./build/task1

SOURCES = timer.c
SRC_TASK_1 = ./task_1/main.c

all: build_task_1

build_task_1: $(SRC_TASK_1) ${SOURCES}
	${CC} $(SRC_TASK_1) $(SOURCES) -o $(TARGET_TASK_1)
