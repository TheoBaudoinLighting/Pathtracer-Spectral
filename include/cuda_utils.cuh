#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)

inline void check_cuda_error(cudaError_t result, const char* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
                  << file << ":" << line << " '" << func << "': "
                  << cudaGetErrorString(result) << std::endl;
        cudaDeviceReset();
        exit(99);
    }
}

struct ImageProperties {
    int width;
    int height;
    int num_pixels;
    int samples_per_pixel;
    int max_depth;
    int spectral_samples;
    float lambda_min;
    float lambda_max;
};

template<typename T>
T* cuda_alloc(size_t count) {
    T* dev_ptr;
    CHECK_CUDA_ERROR(cudaMalloc(&dev_ptr, count * sizeof(T)));
    return dev_ptr;
}

template<typename T>
void cuda_free(T* dev_ptr) {
    if (dev_ptr) {
        CHECK_CUDA_ERROR(cudaFree(dev_ptr));
    }
}

template<typename T>
void cuda_copy_to_device(const T* host_ptr, T* dev_ptr, size_t count) {
    CHECK_CUDA_ERROR(cudaMemcpy(dev_ptr, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void cuda_copy_to_host(T* dev_ptr, T* host_ptr, size_t count) {
    CHECK_CUDA_ERROR(cudaMemcpy(host_ptr, dev_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
}

inline dim3 calculate_grid_size(int num_elements, int block_size) {
    int num_blocks = (num_elements + block_size - 1) / block_size;

    int grid_size_x = std::min(num_blocks, 65535);  
    int grid_size_y = (num_blocks + grid_size_x - 1) / grid_size_x;

    return dim3(grid_size_x, grid_size_y);
}

#endif // CUDA_UTILS_CUH