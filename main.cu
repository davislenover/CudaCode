#include <iostream>
#include <cuda.h>

// __global__ means this function is going to be called by the CPU to run on the GPU
// __device__ means called by the GPU to run on the GPU
__global__ void test_func(int* value) {
    // Add ten to the pointer
    *value += 10;
}

int main() {
    std::cout << "Hello, World!" << std::endl;

    // Declare variables
    // "h_" is memory locations on the "host", i.e., CPU
    // "d_" are memory locations on the "device", i.e., GPU
    int test_number = 12;
    int* d_c;

    // cudaMalloc will allocate memory on the GPU
    // the first argument is the out parameter pointer, pointing to where the memory is on the GPU
    // do NOT dereference d_c on the CPU!
    cudaMalloc((void**)&d_c, sizeof(int));

    // cudaMemcpy can copy memory to and from the CPU to the GPU
    // HostToDevice means copy from CPU to GPU
    // take data at h_c (a pointer which points to memory (an integer) on the CPU) and copy it to the destination memory address on the GPU (d_c)
    cudaMemcpy(d_c, &test_number, sizeof(int), cudaMemcpyHostToDevice);

    // Setup Block/Grid dimensions for the GPU
    // A "Block" is a chunk of threads and a "Grid" contains a certain number of "Blocks"
    dim3 grid_size(1); // (1,1,1), i.e., 1 block
    dim3 block_size(1); // (1,1,1), i.e., 1 thread per block

    void* args[] = {&d_c };
    // Launch the Kernel (i.e., execute on the GPU)
    cudaLaunchKernel((void*)test_func,grid_size,block_size,args,0,NULL);

    // Copy the modified integer from the GPU back to the CPU
    cudaMemcpy(&test_number, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    // Afterward, free GPU memory allocated
    cudaFree(d_c);

    printf("The value received from the GPU is: %d",test_number);

    return 0;
}
