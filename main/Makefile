CC = gcc
CFLAGS = -Wall -DCMAKE_BUILD_TYPE=Release -DCL_TARGET_OPENCL_VERSION=220
LDFLAGS = -lOpenCL

all: scheduler

scheduler: scheduler.c
	$(CC) $(CFLAGS) -o scheduler scheduler.c $(LDFLAGS)

clean:
	rm -f scheduler
