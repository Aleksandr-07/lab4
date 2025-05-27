#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// CPU-версия сложения векторов
void vectorAddCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

// CUDA-ядро для сложения векторов
__global__ void vectorAddCUDA(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1000000;
    size_t size = N * sizeof(float);

    // Инициализация данных на CPU
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C_cpu = new float[N];
    float *h_C_gpu = new float[N];

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX; // [0, 1]
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Выделение памяти на GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Копирование данных на GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Настройка параметров запуска ядра
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Замер времени для GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Запуск ядра
    vectorAddCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);

    // Копирование результата обратно на CPU
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // Замер времени для CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_A, h_B, h_C_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpuTime = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();

    bool error = false;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > 1e-5) {
            error = true;
            break;
        }
    }

    
    std::cout << "CPU Time: " << cpuTime << " ms" << std::endl;
    std::cout << "GPU Time: " << gpuTime << " ms" << std::endl;
    std::cout << "Results are " << (error ? "NOT " : "") << "consistent!" << std::endl;

    // Очистка памяти
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
