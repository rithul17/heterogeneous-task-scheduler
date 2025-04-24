CC = gcc
CFLAGS = -Wall -DCL_TARGET_OPENCL_VERSION=220
LDFLAGS = -lOpenCL -lpthread

TARGET = hetero_scheduler
SRC = main.c
KERNEL_FILE = kernels.cl

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)
	@echo "Build complete. Don't forget to create the kernel file: $(KERNEL_FILE)"

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
