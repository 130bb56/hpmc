#define TILE_SIZE   16
#define BLOCK_SIZE  16
#define PADDING     8
#define CLASS_NUM   10
#include <curand.h>
#include <curand_kernel.h>
/*
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
        float div = 0.0f;
        for (int i = 0; i < output_dim; i++) {
            div += exp(output[row * output_dim + i] - maxval);
        }
        activation[row * output_dim + col] = exp(output[row * output_dim + col] - maxval) / div;
        // printf("Thread(%d,%d): maxval = %f\n", row, col, maxval);
        // printf("Thread(%d,%d): div = %f\n", row, col, div);
    }

}
*/

// TODO: softmax with warp shuffle
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
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ float x_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float w_tile[TILE_SIZE][TILE_SIZE + PADDING];
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
    int row_idx = row * output_dim;
    output[row_idx + col] = out;
    float maxval = output[row_idx];
    // It's guranteed that output_dim = 10 = CLASS_NUM
    #pragma unroll
    for (int i = 1; i < CLASS_NUM; i++) {
        maxval = fmaxf(maxval, output[row_idx + i]);
    }
    float div = exp(output[row_idx] - maxval);
    #pragma unroll
    for (int i = 1; i < CLASS_NUM; i++) {
        div += exp(output[row_idx + i] - maxval);
    }
    activation[row_idx + col] = exp(output[row_idx + col] - maxval) / div;
}

#if 0
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
    __shared__ float w_tile[TILE_SIZE][TILE_SIZE + PADDING];
    if (row >= batch_size || col >= output_dim) {
        return;
    }
    // Y[row, col] = b[col]
    float out = bias[col];
    for (int tile_offset = 0; tile_offset < input_dim; tile_offset += TILE_SIZE) {
        x_tile[ty][tx] = input[row * input_dim + tile_offset + tx];
        w_tile[ty][tx] = weight[(tile_offset + ty) * output_dim + col];
        __syncthreads();

        // Y[row, col] += X[row, :] * W[:, col]
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            out += x_tile[ty][i] * w_tile[i][tx];
        }
        __syncthreads();
    }
    output[row * output_dim + col] = out;
}
#endif

#if 0 
// It cause race condition
__global__ void softmax(int w, int h, float *input, float *output) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < h && col < w) {
        float maxval = input[row * w]; // Initialize to maxval = a[row, 0]
        for (int i = 1; i < w; i++) {
            maxval = fmaxf(maxval, input[row * w + i]);
        }
        float div = 0.0f;
        for (int i = 0; i < w; i++) {
            div += exp(input[row * w + i] - maxval);
        }
        output[row * w + col] = exp(input[row * w + col] - maxval) / div;
    }
}
#endif

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

// consider coalesed memory access: https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimizaton/
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
    __shared__ float w_tile[TILE_SIZE][TILE_SIZE + PADDING];
    
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

/*
__global__ void z_grad(
    const int batch_size,
    const int input_dim,
    const int output_dim,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ dz,
    float *__restrict__ out_dz,
    const float *__restrict__ activation
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < batch_size && col < output_dim) {
        float d = 0.0f;
        for (int i = 0; i < input_dim; i++) {
            float w = weight[i * output_dim + col];
            d += w * dz[row * input_dim + i];
        }
        float act = activation[row * output_dim + col];
        out_dz[row * output_dim + col] = act > 0.0f ? d : 0.0f;
    }
}
*/

__global__ void z_grad(
    const int batch_size,
    const int input_dim,  // 10
    const int output_dim, // 160
    const float *__restrict__ weight,      // (input_dim, output_dim)
    float *__restrict__ dz,                // (batch_size, input_dim)
    float *__restrict__ out_dz,            // (batch_size, output_dim)
    const float *__restrict__ activation   // (batch_size, output_dim)
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // if (row >= batch_size || col >= output_dim) return;
    __shared__ float dz_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float w_tile[TILE_SIZE][TILE_SIZE + PADDING];
    float dZ = 0.0f;
    // dz^(k) = ∂L/∂Z^(k) = (dz^(k+1) @ (W^(k+1))^t) ⊙ 1(a^(k) > 0)
    for (int tile_offset = 0; tile_offset < input_dim; tile_offset += TILE_SIZE) {
        dz_tile[ty][tx] = (tile_offset + tx < input_dim) ? dz[row * input_dim + tile_offset + tx] : 0.0f;
        w_tile[ty][tx] = (tile_offset + ty < input_dim) ? weight[col * input_dim + (tile_offset + ty)] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            dZ += dz_tile[ty][i] * w_tile[i][tx];
        }
        __syncthreads();
    }
    float a = activation[row * output_dim + col];
    out_dz[row * output_dim + col] = (a > 0.0f) ? dZ : 0.0f;  // 1(a^(k) > 0)
}

__global__ void update_layer(
    const int width,
    const int height,
    const int batch_size,
    const float lr,
    float *__restrict__ weight,
    float *__restrict__ bias,
    const float *__restrict__ activation,
    const float *__restrict__ dz
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        float dw = 0.0f;
        float db = 0.0f;
        // W^(k) -= lr * ∂L/∂W^(k) = lr * (A^(k-1))^t * dz^(k)
        // b^(k) -= lr * ∂L/∂b^(k) = lr * sum(dz^(k))
        // where dz^(k) = ∂L/∂Z^(k)
        for (int i = 0; i < batch_size; i++) {
            float a = activation[i * height + row];
            float dZ = dz[i * width + col];
            dw += a * dZ;
            db += dZ;
        }
        weight[row * width + col] -= lr * dw / batch_size;
        atomicAdd(&bias[col], -lr * db / (batch_size * height));
    }
}

// /*
#define WIDTH 10
__global__ void cross_entropy(
    const int width, 
    const int height, 
    const float *__restrict__ y_hat, 
    const float *__restrict__ y, 
    float *__restrict__ output
) {
    // int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) * WIDTH;
    float loss = 0.0f;
    // width = layer[2].cur_dim = 10
    #pragma unroll
    for (int i = 0; i < WIDTH; i++) {
        loss -= y[tid + i] * logf(fmaxf(1e-6, y_hat[tid + i]));
    }
    output[tid / WIDTH] = loss;
}
// */

#if 0
// No logical errors in the reduction itself, but produces different loss values
// Due to fp precision differences in parallel reduction
// Moreover, it's not even faster than the sequential version
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
    shm_loss[col] = (col < width) ? -y[row * width + col] * logf(fmaxf(1e-6, y_hat[row * width + col])) : 0.0f;
    __syncthreads();
    
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        shm_loss[col] += shm_loss[col + offset];
        __syncthreads();
    }
    if (col == 0)   output[row] = shm_loss[0];
    */
    
    // Since col < width, we can safely omit boundary checks below
    // float loss = (col < width) ? -y[row * width + col] * logf(fmaxf(1e-6, y_hat[row * width + col])) : 0.0f;
    float loss = -y[row * width + col] * logf(fmaxf(1e-6, y_hat[row * width + col]));
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        loss += __shfl_down_sync(0xFFFFFFFF, loss, offset);
    }
    if (col == 0) output[row] = loss;
}
#endif

/* // for (2D, 2D) config
__global__ void cross_entropy_softmax_grad(
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

// for (1D, 1D) config
// how to deriving categorical cross entropy and softmax
// https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/
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
        weight[row * width + col] = curand_normal(&state) * sqrtf(2.0f / height);
    }
}

__global__ void init_bias(const int output_dim, float *__restrict__ bias) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    bias[col] = 0.0f;
}
