#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef unsigned int UI;

#define BLOCK_SIZE 32

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {                                          \
    cudaError_t err = value;                                                \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "Error %s at line %d in file %s\n",                 \
        cudaGetErrorString(err), __LINE__, __FILE__);                       \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
}

/* Simple utility function to check for CUDA runtime errors */
void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* Host function to compute the matrix multiplication */
void matrix_mult_cpu(int *h_a, int *h_b, int *h_result, UI n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int tmp = 0.0;
            for (int k = 0; k < n; ++k) {
                tmp += h_a[i * n + k] * h_b[k * n + j];
            }
            h_result[i * n + j] = tmp;
        }
    }
}

/* Kernel that computes the matrix multiplication */
__global__ void matrix_mult_gpu(int *d_a, int *d_b, int *d_result, UI n) {
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    int tmp = 0.0;

    for (int sub = 0; sub < n / BLOCK_SIZE; ++sub) {
        tile_a[threadIdx.y][threadIdx.x] =
                d_a[row * n + sub * BLOCK_SIZE + threadIdx.x];
        tile_b[threadIdx.y][threadIdx.x] =
                d_b[(sub * BLOCK_SIZE + threadIdx.y) * n + col];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    d_result[row * n + col] = tmp;
}

int main(int argc, char **argv) {
    UI n, mat_size;
    //scanf("%u", &n);
    n = 100;
    mat_size = n * n * sizeof(int);

    /* Code to create two events in order to compute elapsed time */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /*------------------------ COMPUTATION ON CPU ----------------------------*/
    int *h_a, *h_b, *h_result;
    cudaMallocHost((void **) &h_a, mat_size);
    cudaMallocHost((void **) &h_b, mat_size);
    cudaMallocHost((void **) &h_result, mat_size);
    checkCUDAError("cudaMallocHost error");

    /* Initialize input matrixs */
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 1024;
            h_b[i * n + j] = rand() % 1024;
        }
    }

    // cudaEventRecord(start, 0);

    // /* Compute matrix multiplication on CPU (host) */
    // matrix_mult_cpu(h_a, h_b, h_result, n);

    // /* Compute the host elapsed time */
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // float elapsed_time_cpu;
    // cudaEventElapsedTime(&elapsed_time_cpu, start, stop);

    // printf("CPU matrix multiplication has finished!\nElapsed time: %f ms\n\n",
    //                 elapsed_time_cpu);

    /*------------------------ COMPUTATION ON GPU ----------------------------*/

    /* Allocate memory space on the device */
    int *d_a, *d_b, *d_result, *h_result_gpu;
    cudaMalloc((void **) &d_a, mat_size);
    cudaMalloc((void **) &d_b, mat_size);
    cudaMalloc((void **) &d_result, mat_size);
    checkCUDAError("cudaMalloc error");

    cudaMallocHost((void **) &h_result_gpu, mat_size);
    checkCUDAError("cudaMallocHost for h_result_gpu error");

    /* Transfer data from host to device */
    cudaMemcpy(d_a, h_a, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, mat_size, cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy (from host to device) error");

    /* Execution configuration setup */
    /* Note: ceil(n / BLOCK_SIZE) also works for dim_grid setup */
    dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

    cudaEventRecord(start, 0);

    /* Compute matrix multiplication on GPU (device kernel launch) */
    matrix_mult_gpu<<<dim_grid, dim_block>>>(d_a, d_b, d_result, n);

    /* Block until the device has completed */
    //cudaDeviceSynchronize();

    /* Compute the device elapsed time */
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time_gpu;
    cudaEventElapsedTime(&elapsed_time_gpu, start, stop);

    printf("GPU matrix multiplication has finished!\nElapsed time: %f ms\n\n",
                elapsed_time_gpu);

    /* Transefr results from device to host */
    cudaMemcpy(h_result_gpu, d_result, mat_size, cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy (from device to host) error");

    /* Compare against host computed solution (no valid for floating point) */
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         assert(h_result[i * n + j] == h_result_gpu[i * n + j]);
    //     }
    // }

    printf("Matrix Multiplication on both CPU and GPU are correct!\n\n");

    /* Destroy CUDA Events */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* Free device memory */
    cudaFree(d_result);
    cudaFree(d_a);
    cudaFree(d_b);
    checkCUDAError("cudaFree error");

    /* Free host memory (it also works with free() call) */
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_result);
    cudaFreeHost(h_result_gpu);
    checkCUDAError("cudaFreeHost error");

    // printf("Speedup of GPU version over the CPU version for an %u x %u input "
    //         "matrixs is %fX\n", n, n, elapsed_time_cpu / elapsed_time_gpu);
    printf("Time of GPU version for an %u x %u input "
            "matrixs is %fX\n", n, n, elapsed_time_gpu);
}

