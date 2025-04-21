// OpenCL kernel for GPU task - adds 10 to each element
__kernel void gpu_task(__global const float* input, 
                       __global float* output,
                       const int size) {
    // Get the global ID of the work item
    int gid = get_global_id(0);
    
    // Ensure we're within bounds
    if (gid < size) {
        // Add 10 to the value from CPU task
        output[gid] = input[gid] + 10.0f;
    }
}
