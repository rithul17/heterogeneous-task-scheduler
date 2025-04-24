/**
 * Advanced Heterogeneous Task Scheduler
 * 
 * This implementation features:
 * - Multiple CPU and GPU tasks with dependencies
 * - A task graph representing dependencies
 * - Dynamic task scheduling based on availability
 * - Pipeline execution for improved throughput
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <CL/cl.h>
#include <time.h>
#include <pthread.h>

#define MAX_TASKS 16
#define MAX_DEPENDENCIES 8
#define MAX_SOURCE_SIZE (0x100000)
#define CHECK_ERROR(err, msg) if (err != CL_SUCCESS) { printf("%s (Error code: %d)\n", msg, err); exit(EXIT_FAILURE); }

// Enums and structures
typedef enum {
    TASK_TYPE_CPU,
    TASK_TYPE_GPU
} TaskType;

typedef enum {
    TASK_STATE_PENDING,
    TASK_STATE_READY,
    TASK_STATE_RUNNING,
    TASK_STATE_COMPLETED,
    TASK_STATE_FAILED
} TaskState;

// Forward declaration
struct Task;

// Function pointer types
typedef void (*CPUTaskFunction)(float* input, float* output, int size);
typedef const char* (*GPUKernelName)();

// Task structure
typedef struct Task {
    int id;
    char name[64];
    TaskType type;
    TaskState state;
    
    // Task dependencies
    int dependency_count;
    struct Task* dependencies[MAX_DEPENDENCIES];
    
    // Input/output
    float* input_data;
    float* output_data;
    int data_size;
    
    // CPU specific
    CPUTaskFunction cpu_function;
    
    // GPU specific
    GPUKernelName kernel_name;
    cl_mem input_buffer;
    cl_mem output_buffer;
    cl_event completion_event;
    
    // For measuring performance
    double execution_time;
} Task;

// OpenCL context
typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
} OpenCLContext;

// Task scheduler
typedef struct {
    Task tasks[MAX_TASKS];
    int task_count;
    OpenCLContext cl;
    pthread_mutex_t mutex;
    bool initialized;
} TaskScheduler;

// Global scheduler
TaskScheduler scheduler;

// CPU Task implementations
void cpu_task_initialize(float* input, float* output, int size) {
    printf("CPU Task: Initializing data with random values\n");
    srand(42); // Fixed seed for reproducibility
    for (int i = 0; i < size; i++) {
        input[i] = (float)(rand() % 10000000000) / 10.0f;
        output[i] = input[i]; // Copy to output as well
    }
}

void cpu_task_filter(float* input, float* output, int size) {
    printf("CPU Task: Applying threshold filter\n");
    for (int i = 0; i < size; i++) {
        // Simple threshold filter
        output[i] = (input[i] > 5.0f) ? input[i] : 0.0f;
    }
}

void cpu_task_normalize(float* input, float* output, int size) {
    printf("CPU Task: Normalizing data\n");
    // Find max value
    float max_val = 0.0f;
    for (int i = 0; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    // Normalize
    if (max_val > 0.0f) {
        for (int i = 0; i < size; i++) {
            output[i] = input[i] / max_val;
        }
    }
}

void cpu_task_analyze(float* input, float* output, int size) {
    printf("CPU Task: Analyzing results\n");
    float sum = 0.0f, avg = 0.0f, variance = 0.0f;
    int nonzero_count = 0;
    
    // Calculate average of non-zero elements
    for (int i = 0; i < size; i++) {
        if (input[i] > 0.0f) {
            sum += input[i];
            nonzero_count++;
        }
    }
    
    if (nonzero_count > 0) {
        avg = sum / nonzero_count;
    }
    
    // Calculate variance
    for (int i = 0; i < size; i++) {
        if (input[i] > 0.0f) {
            float diff = input[i] - avg;
            variance += diff * diff;
        }
    }
    
    if (nonzero_count > 1) {
        variance /= (nonzero_count - 1);
    }
    
    // Store statistics in first elements of output
    output[0] = nonzero_count; // Count
    output[1] = avg;           // Mean
    output[2] = variance;      // Variance
    output[3] = (nonzero_count > 0) ? sum : 0.0f; // Sum
    
    printf("  Analysis: Count=%d, Mean=%.2f, Variance=%.2f, Sum=%.2f\n", 
           nonzero_count, avg, variance, sum);
}

// GPU Kernel names
const char* gpu_kernel_square() { return "square_values"; }
const char* gpu_kernel_add_constant() { return "add_constant"; }
const char* gpu_kernel_smooth() { return "smooth_data"; }

// Read kernel source from file
char* read_kernel_source(const char* filename, size_t* size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open kernel file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    rewind(file);
    
    char* source = (char*)malloc(*size + 1);
    fread(source, 1, *size, file);
    source[*size] = '\0';
    
    fclose(file);
    return source;
}

// Initialize OpenCL environment
void init_opencl(OpenCLContext* cl) {
    cl_int err;
    
    // Get platform
    err = clGetPlatformIDs(1, &cl->platform, NULL);
    CHECK_ERROR(err, "Failed to get OpenCL platform");
    
    // Get device
    err = clGetDeviceIDs(cl->platform, CL_DEVICE_TYPE_GPU, 1, &cl->device, NULL);
    if (err != CL_SUCCESS) {
        printf("GPU device not found, falling back to CPU\n");
        err = clGetDeviceIDs(cl->platform, CL_DEVICE_TYPE_CPU, 1, &cl->device, NULL);
        CHECK_ERROR(err, "Failed to get CPU device");
    }
    
    // Create context
    cl->context = clCreateContext(NULL, 1, &cl->device, NULL, NULL, &err);
    CHECK_ERROR(err, "Failed to create context");
    
    // Create command queue
    #ifdef CL_VERSION_2_0
        cl_queue_properties props[] = {0};
        cl->queue = clCreateCommandQueueWithProperties(cl->context, cl->device, props, &err);
    #else
        cl->queue = clCreateCommandQueue(cl->context, cl->device, 0, &err);
    #endif
    CHECK_ERROR(err, "Failed to create command queue");
    
    // Load and build kernel
    size_t source_size;
    char* source = read_kernel_source("kernels.cl", &source_size);
    
    cl->program = clCreateProgramWithSource(cl->context, 1, (const char**)&source, &source_size, &err);
    CHECK_ERROR(err, "Failed to create program");
    
    err = clBuildProgram(cl->program, 1, &cl->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(cl->program, cl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(cl->program, cl->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build error: %s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }
    
    free(source);
}

// Clean up OpenCL resources
void cleanup_opencl(OpenCLContext* cl) {
    clReleaseProgram(cl->program);
    clReleaseCommandQueue(cl->queue);
    clReleaseContext(cl->context);
}

// Initialize the scheduler
void scheduler_init() {
    if (scheduler.initialized) return;
    
    scheduler.task_count = 0;
    init_opencl(&scheduler.cl);
    pthread_mutex_init(&scheduler.mutex, NULL);
    scheduler.initialized = true;
    
    printf("Heterogeneous task scheduler initialized\n");
}

// Clean up scheduler resources
void scheduler_cleanup() {
    if (!scheduler.initialized) return;
    
    for (int i = 0; i < scheduler.task_count; i++) {
        Task* task = &scheduler.tasks[i];
        free(task->input_data);
        free(task->output_data);
        
        if (task->type == TASK_TYPE_GPU) {
            if (task->input_buffer) clReleaseMemObject(task->input_buffer);
            if (task->output_buffer) clReleaseMemObject(task->output_buffer);
        }
    }
    
    cleanup_opencl(&scheduler.cl);
    pthread_mutex_destroy(&scheduler.mutex);
    scheduler.initialized = false;
    
    printf("Scheduler resources released\n");
}

// Check if all dependencies are completed
bool are_dependencies_met(Task* task) {
    for (int i = 0; i < task->dependency_count; i++) {
        if (task->dependencies[i]->state != TASK_STATE_COMPLETED) {
            return false;
        }
    }
    return true;
}

// Add a task to the scheduler
Task* scheduler_add_task(const char* name, TaskType type, int data_size) {
    pthread_mutex_lock(&scheduler.mutex);
    
    if (scheduler.task_count >= MAX_TASKS) {
        fprintf(stderr, "Maximum task count reached\n");
        pthread_mutex_unlock(&scheduler.mutex);
        return NULL;
    }
    
    Task* task = &scheduler.tasks[scheduler.task_count];
    task->id = scheduler.task_count++;
    strncpy(task->name, name, sizeof(task->name) - 1);
    task->type = type;
    task->state = TASK_STATE_PENDING;
    task->dependency_count = 0;
    task->data_size = data_size;
    
    // Allocate memory for input/output data
    task->input_data = (float*)malloc(sizeof(float) * data_size);
    task->output_data = (float*)malloc(sizeof(float) * data_size);
    
    if (!task->input_data || !task->output_data) {
        fprintf(stderr, "Failed to allocate memory for task data\n");
        pthread_mutex_unlock(&scheduler.mutex);
        return NULL;
    }
    
    // Clear data
    memset(task->input_data, 0, sizeof(float) * data_size);
    memset(task->output_data, 0, sizeof(float) * data_size);
    
    // GPU-specific initialization
    if (type == TASK_TYPE_GPU) {
        task->input_buffer = NULL;
        task->output_buffer = NULL;
    }
    
    pthread_mutex_unlock(&scheduler.mutex);
    return task;
}

// Add dependency between tasks
bool scheduler_add_dependency(Task* dependent_task, Task* dependency) {
    pthread_mutex_lock(&scheduler.mutex);
    
    if (dependent_task->dependency_count >= MAX_DEPENDENCIES) {
        fprintf(stderr, "Maximum dependencies reached for task %s\n", dependent_task->name);
        pthread_mutex_unlock(&scheduler.mutex);
        return false;
    }
    
    dependent_task->dependencies[dependent_task->dependency_count++] = dependency;
    
    pthread_mutex_unlock(&scheduler.mutex);
    return true;
}

// Set CPU task function
void scheduler_set_cpu_function(Task* task, CPUTaskFunction func) {
    if (task->type != TASK_TYPE_CPU) {
        fprintf(stderr, "Task %s is not a CPU task\n", task->name);
        return;
    }
    
    task->cpu_function = func;
}

// Set GPU kernel name
void scheduler_set_gpu_kernel(Task* task, GPUKernelName kernel_name) {
    if (task->type != TASK_TYPE_GPU) {
        fprintf(stderr, "Task %s is not a GPU task\n", task->name);
        return;
    }
    
    task->kernel_name = kernel_name;
}

// Execute a CPU task
void execute_cpu_task(Task* task) {
    clock_t start, end;
    
    printf("Executing CPU task: %s\n", task->name);
    
    // Copy input data from dependencies if any
    if (task->dependency_count > 0) {
        // Use the output of the first dependency as input
        memcpy(task->input_data, task->dependencies[0]->output_data, 
               sizeof(float) * task->data_size);
    }
    
    start = clock();
    
    // Execute the task function
    if (task->cpu_function) {
        task->cpu_function(task->input_data, task->output_data, task->data_size);
    }
    
    end = clock();
    task->execution_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("CPU task %s completed in %.10f seconds\n", task->name, task->execution_time);
    task->state = TASK_STATE_COMPLETED;
}

// Execute a GPU task
void execute_gpu_task(Task* task) {
    cl_int err;
    clock_t start, end;
    
    printf("Executing GPU task: %s\n", task->name);
    
    start = clock();
    
    // Create input buffer with data from dependency if exists
    if (task->dependency_count > 0) {
        // Use the output of the first dependency as input
        task->input_buffer = clCreateBuffer(scheduler.cl.context, 
                                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(float) * task->data_size, 
                                          task->dependencies[0]->output_data, &err);
    } else {
        // No dependency, use task's own input data
        task->input_buffer = clCreateBuffer(scheduler.cl.context, 
                                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(float) * task->data_size, 
                                          task->input_data, &err);
    }
    CHECK_ERROR(err, "Failed to create input buffer");
    
    // Create output buffer
    task->output_buffer = clCreateBuffer(scheduler.cl.context, 
                                       CL_MEM_WRITE_ONLY,
                                       sizeof(float) * task->data_size, 
                                       NULL, &err);
    CHECK_ERROR(err, "Failed to create output buffer");
    
    // Create kernel
    cl_kernel kernel = clCreateKernel(scheduler.cl.program, task->kernel_name(), &err);
    CHECK_ERROR(err, "Failed to create kernel");
    
    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &task->input_buffer);
    CHECK_ERROR(err, "Failed to set kernel arg 0");
    
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &task->output_buffer);
    CHECK_ERROR(err, "Failed to set kernel arg 1");
    
    err = clSetKernelArg(kernel, 2, sizeof(int), &task->data_size);
    CHECK_ERROR(err, "Failed to set kernel arg 2");
    
    // Execute kernel
    size_t global_work_size = task->data_size;
    err = clEnqueueNDRangeKernel(scheduler.cl.queue, kernel, 1, NULL, 
                                &global_work_size, NULL, 0, NULL, &task->completion_event);
    CHECK_ERROR(err, "Failed to enqueue kernel");
    
    // Read results back to host
    err = clEnqueueReadBuffer(scheduler.cl.queue, task->output_buffer, CL_TRUE, 0,
                            sizeof(float) * task->data_size, task->output_data,
                            1, &task->completion_event, NULL);
    CHECK_ERROR(err, "Failed to read output buffer");
    
    // Clean up
    clReleaseKernel(kernel);
    clReleaseMemObject(task->input_buffer);
    clReleaseMemObject(task->output_buffer);
    task->input_buffer = NULL;
    task->output_buffer = NULL;
    
    end = clock();
    task->execution_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("GPU task %s completed in %.10f seconds\n", task->name, task->execution_time);
    task->state = TASK_STATE_COMPLETED;
}

// Run all tasks respecting dependencies
void scheduler_run() {
    bool progress;
    bool all_completed;
    
    printf("\nStarting task execution\n");
    printf("=======================\n");
    
    do {
        progress = false;
        all_completed = true;
        
        for (int i = 0; i < scheduler.task_count; i++) {
            Task* task = &scheduler.tasks[i];
            
            // Skip completed or running tasks
            if (task->state == TASK_STATE_COMPLETED || task->state == TASK_STATE_RUNNING) {
                continue;
            }
            
            all_completed = false;
            
            // Check if task is ready to run
            if (task->state == TASK_STATE_PENDING && are_dependencies_met(task)) {
                task->state = TASK_STATE_RUNNING;
                
                // Execute task based on type
                if (task->type == TASK_TYPE_CPU) {
                    execute_cpu_task(task);
                } else if (task->type == TASK_TYPE_GPU) {
                    execute_gpu_task(task);
                }
                
                progress = true;
            }
        }
        
    } while (progress && !all_completed);
    
    printf("\nTask execution complete\n");
    printf("======================\n");
    
    // Print task execution summary
    printf("\nTask Execution Summary:\n");
    printf("ID | Name                  | Type | Time (s) | State\n");
    printf("---------------------------------------------------\n");
    for (int i = 0; i < scheduler.task_count; i++) {
        Task* task = &scheduler.tasks[i];
        printf("%-2d | %-20s | %-4s | %7.10f | %s\n",
               task->id,
               task->name,
               task->type == TASK_TYPE_CPU ? "CPU" : "GPU",
               task->execution_time,
               task->state == TASK_STATE_COMPLETED ? "Completed" : "Failed");
    }
}

// Print data for debugging
void print_data(const char* label, float* data, int size, int max_display) {
    printf("%s: ", label);
    for (int i = 0; i < size && i < max_display; i++) {
        printf("%.2f ", data[i]);
    }
    if (size > max_display) printf("...");
    printf("\n");
}

int main() {
    // Increased array size from 1024 to 1,000,000
    const int DATA_SIZE = 1000000;
    
    // Initialize scheduler
    scheduler_init();
    
    // Create tasks with dependencies
    // CPU task to initialize data
    Task* init_task = scheduler_add_task("Initialize", TASK_TYPE_CPU, DATA_SIZE);
    scheduler_set_cpu_function(init_task, cpu_task_initialize);
    
    // GPU task to square all values
    Task* square_task = scheduler_add_task("Square Values", TASK_TYPE_GPU, DATA_SIZE);
    scheduler_set_gpu_kernel(square_task, gpu_kernel_square);
    scheduler_add_dependency(square_task, init_task);
    
    // CPU task to filter values
    Task* filter_task = scheduler_add_task("Filter", TASK_TYPE_CPU, DATA_SIZE);
    scheduler_set_cpu_function(filter_task, cpu_task_filter);
    scheduler_add_dependency(filter_task, square_task);
    
    // GPU task to add constant
    Task* add_task = scheduler_add_task("Add Constant", TASK_TYPE_GPU, DATA_SIZE);
    scheduler_set_gpu_kernel(add_task, gpu_kernel_add_constant);
    scheduler_add_dependency(add_task, filter_task);
    
    // GPU task to smooth data
    Task* smooth_task = scheduler_add_task("Smooth Data", TASK_TYPE_GPU, DATA_SIZE);
    scheduler_set_gpu_kernel(smooth_task, gpu_kernel_smooth);
    scheduler_add_dependency(smooth_task, add_task);
    
    // CPU task to normalize data
    Task* norm_task = scheduler_add_task("Normalize", TASK_TYPE_CPU, DATA_SIZE);
    scheduler_set_cpu_function(norm_task, cpu_task_normalize);
    scheduler_add_dependency(norm_task, smooth_task);
    
    // CPU task to analyze results
    Task* analyze_task = scheduler_add_task("Analyze", TASK_TYPE_CPU, DATA_SIZE);
    scheduler_set_cpu_function(analyze_task, cpu_task_analyze);
    scheduler_add_dependency(analyze_task, norm_task);
    
    // Run all tasks
    scheduler_run();
    
    // Display results
    printf("\nData transformations:\n");
    print_data("Initial", init_task->output_data, DATA_SIZE, 10);
    print_data("After Square", square_task->output_data, DATA_SIZE, 10);
    print_data("After Filter", filter_task->output_data, DATA_SIZE, 10);
    print_data("After Add", add_task->output_data, DATA_SIZE, 10);
    print_data("After Smooth", smooth_task->output_data, DATA_SIZE, 10);
    print_data("Final", norm_task->output_data, DATA_SIZE, 10);
    printf("Analysis results: Count=%.0f, Mean=%.10f, Variance=%.10f, Sum=%.10f\n",
           analyze_task->output_data[0], analyze_task->output_data[1],
           analyze_task->output_data[2], analyze_task->output_data[3]);
    
    // Clean up
    scheduler_cleanup();
    
    return 0;
}
