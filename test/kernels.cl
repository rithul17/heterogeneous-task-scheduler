/**
 * OpenCL Kernels for Heterogeneous Task Scheduler Example
 * 
 * This file contains multiple GPU kernels that are used in the example.
 */

// Square all values in the array
__kernel void square_values(__global const float* input, 
                           __global float* output,
                           const int size) {
    // Get global ID
    int gid = get_global_id(0);
    
    // Ensure we're within bounds
    if (gid < size) {
        // Square the value
        output[gid] = input[gid] * input[gid];
    }
}

// Add a constant value to all elements
__kernel void add_constant(__global const float* input, 
                          __global float* output,
                          const int size) {
    // Get global ID
    int gid = get_global_id(0);
    
    // Constant value to add
    const float value_to_add = 3.0f;
    
    // Ensure we're within bounds
    if (gid < size) {
        // Add constant
        output[gid] = input[gid] + value_to_add;
    }
}

// Apply smoothing (moving average filter)
__kernel void smooth_data(__global const float* input, 
                         __global float* output,
                         const int size) {
    // Get global ID
    int gid = get_global_id(0);
    
    // Window size for smoothing (must be odd)
    const int window = 5;
    const int half_window = window / 2;
    
    // Ensure we're within bounds and have enough elements for the window
    if (gid >= half_window && gid < size - half_window) {
        float sum = 0.0f;
        
        // Calculate average of window elements
        for (int i = -half_window; i <= half_window; i++) {
            sum += input[gid + i];
        }
        
        output[gid] = sum / window;
    }
    else if (gid < size) {
        // Edge case - just copy the original value
        output[gid] = input[gid];
    }
}
