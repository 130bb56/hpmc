#include "profile.cuh"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cassert>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
KernelProfiler timer_init_weight("init_weight");
KernelProfiler timer_init_bias("init_bias");
KernelProfiler timer_forward_relu1("forward_relu1");
KernelProfiler timer_forward_relu2("forward_relu2");
KernelProfiler timer_forward_softmax("forward_softmax");
KernelProfiler timer_cross_entropy("cross_entropy");
KernelProfiler timer_cross_entropy_backwards("cross_entropy_backwards");
KernelProfiler timer_backward1("backward1");
KernelProfiler timer_backward2("backward2");
KernelProfiler timer_update_layer1("update_layer1");
KernelProfiler timer_update_layer2("update_layer2");
KernelProfiler timer_update_layer3("update_layer3");
KernelProfiler timer_forward_relu3("forward_relu3");
KernelProfiler timer_forward_relu4("forward_relu4");
KernelProfiler timer_forward_softmax5("forward_softmax5");
KernelProfiler timer_cross_entropy2("cross_entropy2");

#define TILE_SIZE 16

#define CHECK_ERROR(ans) { cudaAssert((ans), __FILE__, __LINE__); }
#define CHECK_KERNEL_ERROR() { cudaKernelAssert(__FILE__, __LINE__); }
#define ASSERT(cond, msg, ...) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "[%s: %d line] " msg "\n\n", __FILE__, __LINE__, ##__VA_ARGS__); \
            assert(cond); \
        } \
    } while (0)

bool debug;

// CUDA Error Handling: Asynchronous vs Synchronous
// https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
// sync error checking
inline void cudaAssert(cudaError_t err, const char *file, const int line, bool abort = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaAssert(): %s\n[%s: %d]\n\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

// sync/async error checking
inline void cudaKernelAssert(const char *file, const int line, bool abort = true) {
    if (debug) CHECK_ERROR(cudaDeviceSynchronize());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaKernelAssert(): %s\n[%s: %d]\n\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

inline int argmax(const float *__restrict__ arr, const int len) {
    int max_idx = 0;
    float max_val = arr[0];
    for (int i = 1; i < len; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}

struct Layer {
    // host
    int prev_dim;
    int cur_dim;
    // device
    float *w;
    float *b;
    float *d_l;
    float *x;
    float *a;
} layer[3];


// TODO: Separate forward and softmax
__global__ void forward_softmax(
    const int batch_size,
    const int input_dim,
    const int output_dim,
    float *input,
    float *weight,
    float *bias,
    float *output,
    float *activation
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < batch_size && col < output_dim) {
        // Y[row, col] = b[col]
        output[row * output_dim + col] = bias[col];
        for (int i = 0; i < input_dim; i++) {
            // Y[row, col] += X[row, :] * W[:, col]
            output[row * output_dim + col] += input[row * input_dim + i] * weight[i * output_dim + col];
        }
        float maxval = output[row * output_dim];
        for (int i = 1; i < output_dim; i++) {
            maxval = fmaxf(maxval, output[row * output_dim + i]);
        }
        float divisor = 0.0f;
        for (int i = 0; i < output_dim; i++) {
            divisor += exp(output[row * output_dim + i] - maxval);
        }
        activation[row * output_dim + col] = exp(output[row * output_dim + col] - maxval) / divisor;
        // printf("Thread(%d,%d): maxval = %f\n", row, col, maxval);
        // printf("Thread(%d,%d): divisor = %f\n", row, col, divisor);
    }

}

// #if 0
__global__ void forward(
    const int batch_size,
    const int input_dim,
    const int output_dim,
    const float *__restrict__ input,
    float *__restrict__ weight,
    float *__restrict__ bias,
    float *__restrict__ output
) {
    /*
        We want to parallelize Y = XW + b
        where
            X : (batch_size, input_dim)
            W : (input_dim, output_dim)
            b : (1, output_dim)
            Y : (batch_size, output_dim)
        This CUDA kernel performs the Row-major matrix multiplication
            Y = XW + b
        Each thread computes a single element Y[row, col].
    */
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ float x_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float w_tile[TILE_SIZE][TILE_SIZE + 4];
    if (row >= batch_size || col >= output_dim) {
        return;
    }
    float out = bias[col];
    for (int tile_offset = 0; tile_offset < input_dim; tile_offset += TILE_SIZE) {
        x_tile[ty][tx] = input[row * input_dim + tile_offset + tx];
        w_tile[ty][tx] = weight[(tile_offset + ty) * output_dim + col];
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            out += x_tile[ty][i] * w_tile[i][tx];
        }
        __syncthreads();
    }
    output[row * output_dim + col] = out;
    /*
    if (row < batch_size && col < output_dim) {
        // Y[row, col] = b[col]
        output[row * output_dim + col] = bias[col];
        for (int i = 0; i < input_dim; i++) {
            // Y[row, col] += X[row, :] * W[:, col]
            output[row * output_dim + col] += input[row * input_dim + i] * weight[i * output_dim + col];
        }
    }
    */
}

/* // It cause race condition
__global__ void softmax(int w, int h, float *input, float *output) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < h && col < w) {
        float maxval = input[row * w]; // Initialize to maxval = a[row, 0]
        for (int i = 1; i < w; i++) {
            maxval = fmaxf(maxval, input[row * w + i]);
        }
        float divisor = 0.0f;
        for (int i = 0; i < w; i++) {
            divisor += exp(input[row * w + i] - maxval);
        }
        output[row * w + col] = exp(input[row * w + col] - maxval) / divisor;
    }
}
*/

// #endif

#if 0
__global__ void forward_relu(
    int batch_size,
    int input_dim,
    int output_dim,
    float *input,
    float *weight,
    float *bias,
    float *output
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < batch_size && col < output_dim) {
        float out = bias[col];
        for (int i = 0; i < input_dim; i++) {
            out += weight[i * output_dim + col] * input[row * input_dim + i];
        }
        output[row * output_dim + col] = out > 0.0f ? out : 0.0f;
    }
}
#endif

// #if 0
__global__ void forward_relu(
    const int batch_size,
    const int input_dim,
    const int output_dim,
    const float *__restrict__ input,
    float *__restrict__ weight,
    float *__restrict__ bias,
    float *__restrict__ output
) {
    __shared__ float x_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float w_tile[TILE_SIZE][TILE_SIZE + 4];
    
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    if (row >= batch_size || col >= output_dim) {
        return;
    }
    float out = bias[col];
    for (int tile_offset = 0; tile_offset < input_dim; tile_offset += TILE_SIZE) {
        // Since input_dim and output_dim are both divisible by TILE_SIZE, we can safely omit boundary checks below
        // x_tile[ty][tx] = tile_offset + tx < input_dim ? input[row * input_dim + tile_offset + tx] : 0.0f;
        // w_tile[ty][tx] = tile_offset + ty < input_dim ? weight[(tile_offset + ty) * output_dim + col] : 0.0f;
        x_tile[ty][tx] = input[row * input_dim + tile_offset + tx];
        w_tile[ty][tx] = weight[(tile_offset + ty) * output_dim + col];
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            out += x_tile[ty][i] * w_tile[i][tx];
        }
        __syncthreads();
    }
    output[row * output_dim + col] = fmaxf(out, 0.0f);
}
// #endif


__global__ void z_grad(
    const int batch_size,
    const int input_dim,
    const int output_dim,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ d_l,
    float *__restrict__ out_d_l,
    const float *__restrict__ activation
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < batch_size && col < output_dim) {
        float dl = 0.0f;
        for (int i = 0; i < input_dim; i++) {
            float w = weight[i * output_dim + col];
            dl += w * d_l[row * input_dim + i];
        }
        float act = activation[row * output_dim + col];
        out_d_l[row * output_dim + col] = act > 0.0f ? dl : 0.0f;
    }
}

// Using single update_layer kernel instead of separate update_weight/update_bias kernels
// reduces runtime by 250ms with only 0.05% accuracy loss
__global__ void update_layer(
    const int width,
    const int height,
    const int batch_size,
    const float lr,
    float *__restrict__ weight,
    float *__restrict__ bias,
    const float *__restrict__ activation,
    const float *__restrict__ d_l
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        float dw = 0.0f;
        float db = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float act = activation[i * height + row];
            float dl = d_l[i * width + col];
            dw += act * dl;
            db += dl;
        }
        weight[row * width + col] -= lr * dw / batch_size;
        atomicAdd(&bias[col], -lr * db / (batch_size * height));
    }
}


/*
__global__ void cross_entropy(int width, int height, float *y_hat, float *y, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < height) {
        float loss = 0.0f;
        for (int i = 0; i < width; i++) {
            loss -= y[idx * width + i] * logf(fmaxf(1e-6, y_hat[idx * width + i]));
        }
        output[idx] = loss;
    }
}
*/

// consider coalesed memory access: https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/
// #if 0
#define BLOCK_SIZE 16
__global__ void cross_entropy(
    const int width, 
    const int height, 
    const float *__restrict__ y_hat, 
    const float *__restrict__ y, 
    float *__restrict__ output
) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    // Block-level reduction from scratch
    /*
    __shared__ float shm_loss[BLOCK_SIZE];
    shm_loss[col] = (col < width ) ? -y[row * width + col] * logf(fmaxf(1e-6, y_hat[row * width + col])) : 0.0f;
    __syncthreads();
    
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (col < offset) {
            shm_loss[col] += shm_loss[col + offset];
        }
        __syncthreads();
    }
    if (col == 0)   output[row] = shm_loss[0];
    */

    // Since col < width, we can safely omit boundary checks below
    // float loss = (col < width) ? -y[row * width + col] * logf(fmaxf(1e-6, y_hat[row * width + col])) : 0.0f;
    float loss = -y[row * width + col] * logf(fmaxf(1e-6, y_hat[row * width + col]));
    for (int offset = 16; offset > 0; offset >>= 1) {
        loss += __shfl_down_sync(0xFFFFFFFF, loss, offset);
    }
    if (col == 0) output[row] = loss;
}
// #endif

/*
__global__ void cross_entropy_backwards(
    const int width, 
    const int height, 
    const float *__restrict__ y_hat, 
    const float *__restrict__ y, 
    float *__restrict__ output
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        output[row * width + col] = y_hat[row * width + col] - y[row * width + col];
    }
}
*/

__global__ void cross_entropy_softmax_grad(
    const int width, 
    const int height, 
    const float *__restrict__ y_hat, 
    const float *__restrict__ y, 
    float *__restrict__ output
) {
    // col is always < height * width
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    output[col] = y_hat[col] - y[col];
}

__global__ void init_weight(const int width, const int height, float *__restrict__ weight) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        // He initialization: https://arxiv.org/pdf/1502.01852.pdf
        curandState state;
        curand_init(20250228, row * width + col, 0, &state);
        weight[row * width + col] = curand_normal(&state) * sqrtf(2.f / height);
    }
}

__global__ void init_bias(const int output_dim, float *__restrict__ bias) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    bias[col] = 0.0f;
}

void init_layer_params(
    float *__restrict__ weight, 
    float *__restrict__ bias, 
    const int width, 
    const int height, 
    const int block_size
) {
    dim3 dimGrid = dim3(ceil(width / (float)block_size), ceil(height / (float)block_size), 1);
    dim3 dimBlock = dim3(block_size, block_size, 1);
    timer_init_weight.start_timing();
    init_weight<<<dimGrid, dimBlock>>>(width, height, weight);
    CHECK_KERNEL_ERROR();
    timer_init_weight.stop_timing();

    dimGrid = dim3(ceil(height / (float)block_size), 1, 1);
    dimBlock = dim3(block_size, 1, 1);
    timer_init_bias.start_timing();
    init_bias<<<dimGrid, dimBlock>>>(width, bias);
    CHECK_KERNEL_ERROR();
    timer_init_bias.stop_timing();
}

void read_dataset(std::ifstream &fin, const char *path, const int start, const int length, float *__restrict__ x, float *__restrict__ y) {
    constexpr int input_size = 784;
    constexpr int labels = 10;

    std::string line;
    std::vector<char> buffer(4096);
    try {
        if (!fin.good()) {
            throw std::runtime_error("File not found: " + std::string(path));
        }
        for (int i = start; i < start + length; i++) {
            if (!std::getline(fin, line)) {
                throw std::runtime_error("Unexpected end of file");
            }
    
            std::istringstream ss(line);
    
            int label;
            if (!(ss >> label)) {
                throw std::runtime_error("Failed to read label");
            }
    
            std::memset(y + labels * i, 0, labels * sizeof(float));
            y[labels * i + label] = 1.0f;
    
            float *x_row = &x[i * input_size];
            for (int j = 0; j < input_size; j++) {
                ASSERT(ss.getline(&buffer[0], buffer.size(), ','), "Failed to read pixel value for entry %d, pixel %d", i, j);
                x_row[j] = std::strtof(&buffer[0], nullptr) / 255.0f;
            }
        }
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        exit(EXIT_FAILURE);
    }
   
}


int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--debug")) {
            debug = true;
            std::cout << "Debug mode set: CUDA Async Error Checking Enabled" << std::endl;
        }
    }
    auto start_time = std::chrono::system_clock::now();
    constexpr int val_length = 10000;
    constexpr int train_length = 60000;
    constexpr int input_size = 784;
    constexpr int num_class = 10;
    constexpr int block_size = 16;
    constexpr int epochs = 30;
    constexpr int batch_size = 64;
    constexpr float lr = 0.03f;
    layer[0].cur_dim = 320;
    layer[1].cur_dim = 160;
    layer[2].cur_dim = 10;

    dim3 dimGrid;
    dim3 dimBlock;
    float *input;
    float *label;

    float *dataset_train_x = (float *)malloc(input_size * train_length * sizeof(float));
    float *dataset_train_y = (float *)malloc(num_class * train_length * sizeof(float));
    float *dataset_val_x = (float *)malloc(input_size * val_length * sizeof(float));
    float *dataset_val_y = (float *)malloc(num_class * val_length * sizeof(float));

    const char *train_path = "../../data/mnist_train.csv";
    const char *val_path = "../../data/mnist_test.csv";
    std::ifstream train_fin(train_path);
    std::ifstream val_fin(val_path);

    float *out_h = (float *)malloc(batch_size * layer[2].cur_dim * sizeof(float));
    float *loss_h = (float *)malloc(batch_size * sizeof(float));
    float *loss_d;

    cudaStream_t stream;
    CHECK_ERROR(cudaStreamCreate(&stream));
    CHECK_ERROR(cudaMallocAsync((void **)&input, input_size * batch_size * sizeof(float), stream));
    CHECK_ERROR(cudaMallocAsync((void **)&label, num_class * batch_size * sizeof(float), stream));
    read_dataset(train_fin, train_path, 0, train_length, dataset_train_x, dataset_train_y);
    read_dataset(val_fin, val_path, 0, val_length, dataset_val_x, dataset_val_y);
    

    for (int i = 0; i < 3; i++) {
        layer[i].prev_dim = i == 0 ? input_size : (i == 1 ? layer[0].cur_dim : layer[1].cur_dim);
        layer[i].cur_dim = i == 0 ? layer[0].cur_dim : (i == 1 ? layer[1].cur_dim : layer[2].cur_dim);
        CHECK_ERROR(cudaMalloc((void **)&layer[i].w, layer[i].prev_dim * layer[i].cur_dim * sizeof(float)));
        CHECK_ERROR(cudaMalloc((void **)&layer[i].b, layer[i].cur_dim * sizeof(float)));
        CHECK_ERROR(cudaMalloc((void **)&layer[i].d_l, batch_size * layer[i].cur_dim * sizeof(float)));
        init_layer_params(layer[i].w, layer[i].b, layer[i].cur_dim, layer[i].prev_dim, block_size);
        CHECK_ERROR(cudaMalloc((void **)&layer[i].x, batch_size * layer[i].cur_dim * sizeof(float)));
        CHECK_ERROR(cudaMalloc((void **)&layer[i].a, batch_size * layer[i].cur_dim * sizeof(float)));
    }
    CHECK_ERROR(cudaMalloc((void **)&loss_d, batch_size * sizeof(float)));

    float training_time = 0.0f;
    for (int epoch = 1; epoch <= epochs; epoch++) {
        float cum_loss = 0.0f;
        int train_correct = 0;
        int train_cnt = 0;
        auto start_time = std::chrono::system_clock::now();
        // We ignore last batch
        for (int batch_idx = 0; batch_idx < train_length / batch_size; batch_idx++) {
            train_cnt += batch_size;
            CHECK_ERROR(cudaMemcpy(input, &dataset_train_x[batch_idx * batch_size * input_size], batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_ERROR(cudaMemcpy(label, &dataset_train_y[batch_idx * batch_size * num_class], batch_size * num_class * sizeof(float), cudaMemcpyHostToDevice));
            dimGrid = dim3(ceil(layer[0].cur_dim / (float)block_size), ceil(batch_size / (float)block_size), 1);
            dimBlock = dim3(block_size, block_size, 1);
            timer_forward_relu1.start_timing();
            forward_relu<<<dimGrid, dimBlock>>>(
                batch_size, 
                layer[0].prev_dim, 
                layer[0].cur_dim, 
                input, 
                layer[0].w, 
                layer[0].b, 
                layer[0].a
            ); CHECK_KERNEL_ERROR();
            timer_forward_relu1.stop_timing();

            dimGrid = dim3(ceil(layer[1].cur_dim / (float)block_size), ceil(batch_size / (float)block_size), 1);
            timer_forward_relu2.start_timing();
            forward_relu<<<dimGrid, dimBlock>>>(
                batch_size, 
                layer[1].prev_dim, 
                layer[1].cur_dim, 
                layer[0].a, 
                layer[1].w, 
                layer[1].b, 
                layer[1].a
            ); CHECK_KERNEL_ERROR();
            timer_forward_relu2.stop_timing();

            dimGrid = dim3(ceil(layer[2].cur_dim / (float)block_size), ceil(batch_size / (float)block_size), 1);
            /*
            forward<<<dimGrid, dimBlock>>>(
                batch_size, 
                layer[2].prev_dim, 
                layer[2].cur_dim, 
                layer[1].a, 
                layer[2].w, 
                layer[2].b, 
                layer[2].x
            ); CHECK_KERNEL_ERROR();
            softmax<<<dimGrid, dimBlock>>>(layer[2].cur_dim, batch_size, layer[2].x, layer[2].a);
            CHECK_KERNEL_ERROR();
            */
            timer_forward_softmax.start_timing();
            forward_softmax<<<dimGrid, dimBlock>>>(
                batch_size, 
                layer[2].prev_dim, 
                layer[2].cur_dim, 
                layer[1].a, 
                layer[2].w, 
                layer[2].b, 
                layer[2].x, 
                layer[2].a
            ); CHECK_KERNEL_ERROR();
            timer_forward_softmax.stop_timing();
            dimGrid = dim3(ceil(batch_size / (float)block_size), 1, 1);
            dimBlock = dim3(block_size, 1, 1);
            timer_cross_entropy.start_timing();
            cross_entropy<<<dimGrid, dimBlock>>>(layer[2].cur_dim, batch_size, layer[2].a, label, loss_d);
            CHECK_KERNEL_ERROR();
            timer_cross_entropy.stop_timing();

            // TODO: Config layout: (1D, 1D) vs (2D, 2D)
            /*
            dimGrid = dim3(ceil(layer[2].cur_dim / (float)block_size), ceil(batch_size / (float)block_size), 1);
            dimBlock = dim3(block_size, block_size, 1);
            */
            dimGrid = dim3(ceil(layer[2].cur_dim * batch_size / (float)block_size), 1, 1);
            dimBlock = dim3(block_size, 1, 1);
            timer_cross_entropy_backwards.start_timing();
            cross_entropy_softmax_grad<<<dimGrid, dimBlock>>>(layer[2].cur_dim, batch_size, layer[2].a, label, layer[2].d_l);
            CHECK_KERNEL_ERROR();
            timer_cross_entropy_backwards.stop_timing();

            dimGrid = dim3(ceil(layer[1].cur_dim / (float)block_size), ceil(batch_size / (float)block_size), 1);
            dimBlock = dim3(block_size, block_size, 1);
            timer_backward1.start_timing();
            z_grad<<<dimGrid, dimBlock>>>(
                batch_size, 
                layer[2].cur_dim, 
                layer[1].cur_dim, 
                layer[2].w, 
                layer[2].b, 
                layer[2].d_l, 
                layer[1].d_l, 
                layer[1].a
            ); CHECK_KERNEL_ERROR();
            timer_backward1.stop_timing();

            dimGrid = dim3(ceil(layer[0].cur_dim / (float)block_size), ceil(batch_size / (float)block_size), 1);
            timer_backward2.start_timing();
            z_grad<<<dimGrid, dimBlock>>>(
                batch_size, 
                layer[1].cur_dim, 
                layer[0].cur_dim, 
                layer[1].w, 
                layer[1].b, 
                layer[1].d_l, 
                layer[0].d_l, 
                layer[0].a
            ); CHECK_KERNEL_ERROR();
            timer_backward2.stop_timing();

            dimGrid = dim3(ceil(layer[2].cur_dim / (float)block_size), ceil(layer[1].cur_dim / (float)block_size), 1);
            timer_update_layer1.start_timing();
            update_layer<<<dimGrid, dimBlock>>>(
                layer[2].cur_dim, 
                layer[1].cur_dim, 
                batch_size,
                lr, 
                layer[2].w, 
                layer[2].b, 
                layer[1].a, 
                layer[2].d_l
            ); CHECK_KERNEL_ERROR();
            timer_update_layer1.stop_timing();

            dimGrid = dim3(ceil(layer[1].cur_dim / (float)block_size), ceil(layer[0].cur_dim / (float)block_size), 1);
            timer_update_layer2.start_timing();
            update_layer<<<dimGrid, dimBlock>>>(
                layer[1].cur_dim, 
                layer[0].cur_dim, 
                batch_size, 
                lr, 
                layer[1].w, 
                layer[1].b, 
                layer[0].a, 
                layer[1].d_l
            ); CHECK_KERNEL_ERROR();
            timer_update_layer2.stop_timing();

            dimGrid = dim3(ceil(layer[0].cur_dim / (float)block_size), ceil(input_size / (float)block_size), 1);
            timer_update_layer3.start_timing();
            update_layer<<<dimGrid, dimBlock>>>(
                layer[0].cur_dim, 
                input_size, 
                batch_size, 
                lr, 
                layer[0].w, 
                layer[0].b, 
                input, 
                layer[0].d_l
            ); CHECK_KERNEL_ERROR();
            timer_update_layer3.stop_timing();
            CHECK_ERROR(cudaMemcpy(out_h, layer[2].a, batch_size * layer[2].cur_dim * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_ERROR(cudaMemcpy(loss_h, loss_d, batch_size * sizeof(float), cudaMemcpyDeviceToHost));

            for (int i = 0; i < batch_size; i++) {
                int y_hat = argmax(out_h + i * num_class, num_class);
                int y     = argmax(dataset_train_y + (batch_idx * batch_size + i) * num_class, num_class);
                train_correct += (y_hat == y);
                cum_loss += loss_h[i];
            }
        }
        float val_loss = 0.0f;
        int val_correct = 0;
        int val_cnt = 0;
        for (int batch_idx = 0; batch_idx < val_length / batch_size; batch_idx++) {
            val_cnt += batch_size;
            CHECK_ERROR(cudaMemcpy(input, &dataset_val_x[batch_idx * batch_size * input_size], batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_ERROR(cudaMemcpy(label, &dataset_val_y[batch_idx * batch_size * num_class], batch_size * num_class * sizeof(float), cudaMemcpyHostToDevice));

            dimGrid = dim3(ceil(layer[0].cur_dim / (float)block_size), ceil(batch_size / (float)block_size), 1);
            dimBlock = dim3(block_size, block_size, 1);
            timer_forward_relu3.start_timing();
            forward_relu<<<dimGrid, dimBlock>>>(
                batch_size, 
                input_size, 
                layer[0].cur_dim, 
                input, 
                layer[0].w, 
                layer[0].b, 
                layer[0].a
            ); CHECK_KERNEL_ERROR();
            timer_forward_relu3.stop_timing();

            dimGrid = dim3(ceil(layer[1].cur_dim / (float)block_size), ceil(batch_size / (float)block_size), 1);
            dimBlock = dim3(block_size, block_size, 1);
            timer_forward_relu4.start_timing();
            forward_relu<<<dimGrid, dimBlock>>>(
                batch_size, 
                layer[0].cur_dim, 
                layer[1].cur_dim, 
                layer[0].a, 
                layer[1].w, 
                layer[1].b, 
                layer[1].a
            ); CHECK_KERNEL_ERROR();
            timer_forward_relu4.stop_timing();

            dimGrid = dim3(ceil(layer[2].cur_dim / (float)block_size), ceil(batch_size / (float)block_size), 1);
            dimBlock = dim3(block_size, block_size, 1);
            /*
            forward<<<dimGrid, dimBlock>>>(
                batch_size, 
                layer[1].cur_dim,
                layer[2].cur_dim,
                layer[1].a,
                layer[2].w,
                layer[2].b,
                layer[2].x
            ); CHECK_KERNEL_ERROR();
            softmax<<<dimGrid, dimBlock>>>(layer[2].cur_dim, batch_size, layer[2].x, layer[2].a);
            CHECK_KERNEL_ERROR();
            */
            timer_forward_softmax5.start_timing();
            forward_softmax<<<dimGrid, dimBlock>>>(
                batch_size, 
                layer[1].cur_dim,
                layer[2].cur_dim,
                layer[1].a,
                layer[2].w,
                layer[2].b,
                layer[2].x,
                layer[2].a 
            ); CHECK_KERNEL_ERROR();
            timer_forward_softmax5.stop_timing();
            dimGrid = dim3(ceil(batch_size / (float)block_size), 1, 1);
            dimBlock = dim3(block_size, 1, 1);
            timer_cross_entropy2.start_timing();
            cross_entropy<<<dimGrid, dimBlock>>>(layer[2].cur_dim, batch_size, layer[2].a, label, loss_d);
            timer_cross_entropy2.stop_timing();

            CHECK_ERROR(cudaDeviceSynchronize());
            CHECK_ERROR(cudaMemcpy(out_h, layer[2].a, batch_size * layer[2].cur_dim * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_ERROR(cudaMemcpy(loss_h, loss_d, batch_size * sizeof(float), cudaMemcpyDeviceToHost));

            for (int i = 0; i < batch_size; i++) {
                int y_hat = argmax(&out_h[i * num_class], num_class);
                int y = argmax(&dataset_val_y[batch_idx * batch_size * num_class + i * num_class], num_class);
                val_correct += (y == y_hat);
                val_loss    += loss_h[i];
            }
        }

        float epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count();
        training_time += epoch_time;

        printf(
            "epoch %d: %.0fms train loss: %f train accuracy: %f val loss: %f val accuracy: %f\n",
            epoch, epoch_time, cum_loss / train_cnt, (float)train_correct / train_cnt, val_loss / val_cnt, (float)val_correct / val_cnt
        );
    }
    printf("Total Training Time: %.0fms\n", training_time);
    printf("%.0fms per epoch\n", training_time / epochs);
    // int runtime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count();
    // printf("Runtime: %dms\n", runtime);
    for (int i = 0; i < 3; i++) {
        CHECK_ERROR(cudaFree(layer[i].w));
        CHECK_ERROR(cudaFree(layer[i].b));
        CHECK_ERROR(cudaFree(layer[i].d_l));
        CHECK_ERROR(cudaFree(layer[i].x));
        CHECK_ERROR(cudaFree(layer[i].a));
    }
    CHECK_ERROR(cudaFree(input));
    CHECK_ERROR(cudaFree(label));
    CHECK_ERROR(cudaFree(loss_d));
    CHECK_ERROR(cudaStreamDestroy(stream));
    free(dataset_train_x);
    free(dataset_train_y);
    free(dataset_val_x);
    free(dataset_val_y);
    free(out_h);
    free(loss_h);
	KernelProfiler::print();
}
