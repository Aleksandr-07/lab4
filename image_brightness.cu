#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <random>

// CPU-версия увеличения яркости
void increaseBrightnessCPU(unsigned char* input, unsigned char* output, int width, int height, int delta) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            int new_value = input[idx] + delta;
            output[idx] = (new_value > 255) ? 255 : static_cast<unsigned char>(new_value);
        }
    }
}

// CUDA-ядро для увеличения яркости (2D-грид)
__global__ void increaseBrightnessCUDA(unsigned char* input, unsigned char* output, int width, int height, int delta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int new_value = input[idx] + delta;
        output[idx] = (new_value > 255) ? 255 : static_cast<unsigned char>(new_value);
    }
}

int main() {
    const int WIDTH = 1024;
    const int HEIGHT = 1024;
    const int DELTA = 50;
    const int SIZE = WIDTH * HEIGHT;

    // Инициализация данных на CPU
    unsigned char *h_input = new unsigned char[SIZE];
    unsigned char *h_output_cpu = new unsigned char[SIZE];
    unsigned char *h_output_gpu = new unsigned char[SIZE];

    // Заполнение случайными значениями [0, 255]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (int i = 0; i < SIZE; ++i) {
        h_input[i] = static_cast<unsigned char>(dis(gen));
    }

    // Выделение памяти на GPU
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, SIZE * sizeof(unsigned char));
    cudaMalloc(&d_output, SIZE * sizeof(unsigned char));

    // Копирование данных на GPU
    cudaMemcpy(d_input, h_input, SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Настройка параметров запуска ядра (2D)
    dim3 blockDim(32, 32);  // Блок 32x32 = 1024 потока
    dim3 gridDim(
        (WIDTH + blockDim.x - 1) / blockDim.x,
        (HEIGHT + blockDim.y - 1) / blockDim.y
    );

    // Замер времени для GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Запуск ядра
    increaseBrightnessCUDA<<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT, DELTA);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);

    // Копирование результата обратно на CPU
    cudaMemcpy(h_output_gpu, d_output, SIZE * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Замер времени для CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    increaseBrightnessCPU(h_input, h_output_cpu, WIDTH, HEIGHT, DELTA);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpuTime = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();

    // Проверка корректности
    bool error = false;
    for (int i = 0; i < SIZE; ++i) {
        if (h_output_cpu[i] != h_output_gpu[i]) {
            error = true;
            break;
        }
    }

    
    std::cout << "CPU Time: " << cpuTime << " ms" << std::endl;
    std::cout << "GPU Time: " << gpuTime << " ms" << std::endl;
    std::cout << "Results are " << (error ? "NOT " : "") << "consistent!" << std::endl;

    // Очистка памяти
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
