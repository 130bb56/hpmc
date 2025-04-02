#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include "./include/kernel.cuh"
#include "./include/utils.cuh"

#define CHECK_ERROR(ans) { cudaAssert((ans), __FILE__, __LINE__); }
#define CHECK_KERNEL_ERROR() { cudaKernelAssert(__FILE__, __LINE__); }

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



void init_layer_params(
    float *__restrict__ weight, 
    float *__restrict__ bias, 
    const int width, 
    const int height, 
    const int block_size
) {
    dim3 dimGrid = dim3(ceilf(width / (float)block_size), ceilf(height / (float)block_size), 1);
    dim3 dimBlock = dim3(block_size, block_size, 1);
    init_weight<<<dimGrid, dimBlock>>>(width, height, weight);
    CHECK_KERNEL_ERROR();

    dimGrid = dim3(ceilf(height / (float)block_size), 1, 1);
    dimBlock = dim3(block_size, 1, 1);
    init_bias<<<dimGrid, dimBlock>>>(width, bias);
    CHECK_KERNEL_ERROR();
}

int main(int argc, char **argv) {
    constexpr int val_length = 10000;
    constexpr int train_length = 60000;
    constexpr int input_size = 784;
    constexpr int num_class = 10;
    constexpr int block_size = 16;
    constexpr int epochs = 30;
    constexpr float lr = 0.03f;
    
    // CLI args (default)
    int batch_size = 64;
    int layer1_dim = 320;
    int layer2_dim = 160;
    int layer3_dim = 10;

    // Parse cli args
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--debug")) {
            debug = true;
            std::cout << "Debug mode set: CUDA Async Error Checking Enabled" << std::endl;
        }
        if (!strncmp(argv[i], "--batch_size=", 13)) {
            batch_size = std::stoi(argv[i] + 13);
        }
        if (!strncmp(argv[i], "--layer", 7)) {
            if (!strncmp(argv[i] + 7, "1_dim=", 6)) {
                layer1_dim = std::stoi(argv[i] + 13);
            } else if (!strncmp(argv[i] + 7, "2_dim=", 6)) {
                layer2_dim = std::stoi(argv[i] + 13);
            } else if (!strncmp(argv[i] + 7, "3_dim=", 6)) {
                layer3_dim = std::stoi(argv[i] + 13);
            } else {
                std::cout << "Bad args\n";
                exit(EXIT_FAILURE);
            }
        }
    }
    // printf("batch_size = %d, layer_dim 1, 2, 3: %d, %d, %d\n", batch_size, layer1_dim, layer2_dim, layer2_dim);
    auto start_time = std::chrono::system_clock::now();
    // TODO: Add cli args for --batch_size


    dim3 dimGrid;
    dim3 dimBlock;
    float *input;
    float *label;

    float *dataset_train_x = (float *)malloc(input_size * train_length * sizeof(float));
    float *dataset_train_y = (float *)malloc(num_class * train_length * sizeof(float));
    float *dataset_val_x = (float *)malloc(input_size * val_length * sizeof(float));
    float *dataset_val_y = (float *)malloc(num_class * val_length * sizeof(float));

    const char *train_path = "../data/mnist_train.csv";
    const char *val_path = "../data/mnist_test.csv";
    // std::ifstream train_fin(train_path);
    // std::ifstream val_fin(val_path);

    float *out_h = (float *)malloc(batch_size * layer3_dim * sizeof(float));
    float *loss_h = (float *)malloc(batch_size * sizeof(float));
    float *loss_d;

    cudaStream_t stream;
    CHECK_ERROR(cudaStreamCreate(&stream));
    CHECK_ERROR(cudaMallocAsync((void **)&input, input_size * batch_size * sizeof(float), stream));
    CHECK_ERROR(cudaMallocAsync((void **)&label, num_class * batch_size * sizeof(float), stream));
    read_dataset(train_path, 0, train_length, dataset_train_x, dataset_train_y);
    read_dataset(val_path, 0, val_length, dataset_val_x, dataset_val_y);

    for (int i = 1; i <= 3; i++) {
        int prev_dim = i == 1 ? input_size : (i == 2 ? layer1_dim : layer2_dim);
        int cur_dim = i == 1 ? layer1_dim : (i == 2 ? layer2_dim : layer3_dim);
        CHECK_ERROR(cudaMalloc((void **)&layer[i].w, prev_dim * cur_dim * sizeof(float)));
        CHECK_ERROR(cudaMalloc((void **)&layer[i].b, cur_dim * sizeof(float)));
        CHECK_ERROR(cudaMalloc((void **)&layer[i].dz, batch_size * cur_dim * sizeof(float)));
        init_layer_params(layer[i].w, layer[i].b, cur_dim, prev_dim, block_size);
        CHECK_ERROR(cudaMalloc((void **)&layer[i].x, batch_size * cur_dim * sizeof(float)));
        CHECK_ERROR(cudaMalloc((void **)&layer[i].a, batch_size * cur_dim * sizeof(float)));
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
            /*
            // just for debug
            if (epoch == 1 && batch_idx == 600) {
                for (int j = 0; j < batch_size; j++) {
                    std::cout << "sample_idx: " << j << std::endl;
                    for (int i = 0; i < input_size; i++) {
                        if (i % 9 == 0 && i != 0) std::cout << std::endl;
                        std::cout << dataset_train_x[batch_idx * batch_size * input_size + j * input_size + i] << ", ";
                    }
                    std::cout << std::endl;
                    // one-hot to label
                    std::cout << "label: ";
                    for (int c = 0; c < num_class; ++c) {
                        if (dataset_train_y[(batch_idx * batch_size + j) * num_class + c] == 1.0f) {
                            std::cout << c << std::endl;
                            break;
                        }
                    }
                    std::cout << std::endl;
                }
            }
            */
            train_cnt += batch_size;
            CHECK_ERROR(cudaMemcpy(input, &dataset_train_x[batch_idx * batch_size * input_size], batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_ERROR(cudaMemcpy(label, &dataset_train_y[batch_idx * batch_size * num_class], batch_size * num_class * sizeof(float), cudaMemcpyHostToDevice));
            dimGrid = dim3(_ceil(layer1_dim, block_size), _ceil(batch_size, block_size), 1);
            dimBlock = dim3(block_size, block_size, 1);
            forward_relu<<<dimGrid, dimBlock>>>(
                batch_size, 
                input_size, 
                layer1_dim, 
                input, 
                layer[1].w, 
                layer[1].b, 
                layer[1].a
            ); CHECK_KERNEL_ERROR();

            dimGrid = dim3(_ceil(layer2_dim, block_size), _ceil(batch_size, block_size), 1);
            forward_relu<<<dimGrid, dimBlock>>>(
                batch_size, 
                layer1_dim, 
                layer2_dim, 
                layer[1].a, 
                layer[2].w, 
                layer[2].b, 
                layer[2].a
            ); CHECK_KERNEL_ERROR();

            dimGrid = dim3(_ceil(layer3_dim, block_size), _ceil(batch_size, block_size), 1);
            /*
            forward<<<dimGrid, dimBlock>>>(
                batch_size, 
                layer2_dim, 
                layer3_dim, 
                layer[1].a, 
                layer[2].w, 
                layer[2].b, 
                layer[2].x
            ); CHECK_KERNEL_ERROR();
            softmax<<<dimGrid, dimBlock>>>(layer3_dim, batch_size, layer[2].x, layer[2].a);
            CHECK_KERNEL_ERROR();
            */
            forward_softmax<<<dimGrid, dimBlock>>>(
                batch_size, 
                layer2_dim, 
                layer3_dim, 
                layer[2].a, 
                layer[3].w, 
                layer[3].b, 
                layer[3].x, 
                layer[3].a
            ); CHECK_KERNEL_ERROR();
            dimGrid = dim3(_ceil(batch_size, block_size), 1, 1);
            dimBlock = dim3(block_size, 1, 1);
            cross_entropy<<<dimGrid, dimBlock>>>(
                layer3_dim, batch_size, layer[3].a, label, loss_d
            ); CHECK_KERNEL_ERROR();

            // TODO: Config layout: (1D, 1D) vs (2D, 2D) -> (1D, 1D) is much faster
            dimGrid = dim3(_ceil(layer3_dim * batch_size, block_size), 1, 1);
            cross_entropy_softmax_grad<<<dimGrid, dimBlock>>>(layer3_dim, batch_size, layer[3].a, label, layer[3].dz);
            CHECK_KERNEL_ERROR();

            dimGrid = dim3(_ceil(layer2_dim, block_size), _ceil(batch_size, block_size), 1);
            dimBlock = dim3(block_size, block_size, 1);
            z_grad<<<dimGrid, dimBlock>>>(
                batch_size, 
                layer3_dim, 
                layer2_dim, 
                layer[3].w, 
                layer[3].dz, 
                layer[2].dz, 
                layer[2].a
            ); CHECK_KERNEL_ERROR();

            dimGrid = dim3(_ceil(layer1_dim, block_size), _ceil(batch_size, block_size), 1);
            z_grad<<<dimGrid, dimBlock>>>(
                batch_size, 
                layer2_dim, 
                layer1_dim, 
                layer[2].w, 
                layer[2].dz, 
                layer[1].dz, 
                layer[1].a
            ); CHECK_KERNEL_ERROR();

            dimGrid = dim3(_ceil(layer3_dim, block_size), _ceil(layer2_dim, block_size), 1);
            update_layer<<<dimGrid, dimBlock>>>(
                layer3_dim, 
                layer2_dim, 
                batch_size,
                lr, 
                layer[3].w, 
                layer[3].b, 
                layer[2].a, 
                layer[3].dz
            ); CHECK_KERNEL_ERROR();

            dimGrid = dim3(_ceil(layer2_dim, block_size), _ceil(layer1_dim, block_size), 1);
            update_layer<<<dimGrid, dimBlock>>>(
                layer2_dim, 
                layer1_dim, 
                batch_size, 
                lr, 
                layer[2].w, 
                layer[2].b, 
                layer[1].a, 
                layer[2].dz
            ); CHECK_KERNEL_ERROR();

            dimGrid = dim3(_ceil(layer1_dim, block_size), _ceil(input_size, block_size), 1);
            update_layer<<<dimGrid, dimBlock>>>(
                layer1_dim, 
                input_size, 
                batch_size, 
                lr, 
                layer[1].w, 
                layer[1].b, 
                input, 
                layer[1].dz
            ); CHECK_KERNEL_ERROR();
            CHECK_ERROR(cudaMemcpy(out_h, layer[3].a, batch_size * layer3_dim * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_ERROR(cudaMemcpy(loss_h, loss_d, batch_size * sizeof(float), cudaMemcpyDeviceToHost));

            for (int i = 0; i < batch_size; i++) {
                int y_hat     = argmax(out_h + i * num_class, num_class);
                int y         = argmax(dataset_train_y + (batch_idx * batch_size + i) * num_class, num_class);
                train_correct += (y_hat == y);
                cum_loss      += loss_h[i];
            }
        }
        float val_loss = 0.0f;
        int val_correct = 0;
        int val_cnt = 0;
        for (int batch_idx = 0; batch_idx < val_length / batch_size; batch_idx++) {
            val_cnt += batch_size;
            CHECK_ERROR(cudaMemcpy(input, &dataset_val_x[batch_idx * batch_size * input_size], batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_ERROR(cudaMemcpy(label, &dataset_val_y[batch_idx * batch_size * num_class], batch_size * num_class * sizeof(float), cudaMemcpyHostToDevice));

            dimGrid = dim3(_ceil(layer1_dim, block_size), _ceil(batch_size, block_size), 1);
            dimBlock = dim3(block_size, block_size, 1);
            forward_relu<<<dimGrid, dimBlock>>>(
                batch_size, 
                input_size, 
                layer1_dim, 
                input, 
                layer[1].w, 
                layer[1].b, 
                layer[1].a
            ); CHECK_KERNEL_ERROR();

            dimGrid = dim3(_ceil(layer2_dim, block_size), _ceil(batch_size, block_size), 1);
            dimBlock = dim3(block_size, block_size, 1);
            forward_relu<<<dimGrid, dimBlock>>>(
                batch_size, 
                layer1_dim, 
                layer2_dim, 
                layer[1].a, 
                layer[2].w, 
                layer[2].b, 
                layer[2].a
            ); CHECK_KERNEL_ERROR();

            dimGrid = dim3(_ceil(layer3_dim, block_size), _ceil(batch_size, block_size), 1);
            dimBlock = dim3(block_size, block_size, 1);
            /*
            forward<<<dimGrid, dimBlock>>>(
                batch_size, 
                layer2_dim,
                layer3_dim,
                layer[2].a,
                layer[3].w,
                layer[3].b,
                layer[3].x
            ); CHECK_KERNEL_ERROR();
            softmax<<<dimGrid, dimBlock>>>(layer3_dim, batch_size, layer[3].x, layer[3].a);
            CHECK_KERNEL_ERROR();
            */
            forward_softmax<<<dimGrid, dimBlock>>>(
                batch_size, 
                layer2_dim,
                layer3_dim,
                layer[2].a,
                layer[3].w,
                layer[3].b,
                layer[3].x,
                layer[3].a 
            ); CHECK_KERNEL_ERROR();
            dimGrid = dim3(_ceil(batch_size, block_size), 1, 1);
            dimBlock = dim3(block_size, 1, 1);
            cross_entropy<<<dimGrid, dimBlock>>>(layer3_dim, batch_size, layer[3].a, label, loss_d);

            CHECK_ERROR(cudaDeviceSynchronize());
            CHECK_ERROR(cudaMemcpy(out_h, layer[3].a, batch_size * layer3_dim * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_ERROR(cudaMemcpy(loss_h, loss_d, batch_size * sizeof(float), cudaMemcpyDeviceToHost));

            for (int i = 0; i < batch_size; i++) {
                int y_hat   = argmax(&out_h[i * num_class], num_class);
                int y       = argmax(&dataset_val_y[batch_idx * batch_size * num_class + i * num_class], num_class);
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
    for (int i = 1; i <= 3; i++) {
        CHECK_ERROR(cudaFree(layer[i].w));
        CHECK_ERROR(cudaFree(layer[i].b));
        CHECK_ERROR(cudaFree(layer[i].dz));
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
}