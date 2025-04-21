#include <stdio.h>
#include <stdlib.h>

// Structure to hold task data
typedef struct {
    float* input;
    float* output;
    int size;
} CPUTaskData;

// CPU task function - a simple vector multiplication by 2
void cpu_task(CPUTaskData* data) {
    printf("CPU Task: Processing %d elements\n", data->size);
    
    for (int i = 0; i < data->size; i++) {
        data->output[i] = data->input[i] * 2.0f;
    }
    
    printf("CPU Task: Completed\n");
}
