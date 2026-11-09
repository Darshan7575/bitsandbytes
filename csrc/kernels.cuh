// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <float.h>
#include <ops.cuh>

#ifndef kernels
#define kernels

template <typename T, int BLOCK_SIZE, int NUM_PER_TH, int STOCHASTIC, int DATA_TYPE>
__global__ void kQuantizeBlockwise(
    float* code, T* __restrict__ const A, float* absmax, unsigned char* out, float* __restrict__ const rand,
    const int rand_offset, const int n
);
template <typename T, int QBLOCK_SIZE, int DATA_TYPE>
__global__ void kQuantizeBlockwiseSmall(
    float* code, T* __restrict__ const A, float* absmax, unsigned char* out, float* __restrict__ const rand,
    const int rand_offset, const int n
);
template <typename T, int BLOCK_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE>
__global__ void
    kDequantizeBlockwise(float* code, unsigned char* A, float* absmax, T* out, const int blocksize, const int n);

template <typename T, int OPTIMIZER, int BLOCK_SIZE, int NUM_VALS>
__global__ void kPreconditionOptimizer32bit2State(
    T* g, T* p, float* state1, float* state2, float* unorm, const float beta1, const float beta2, const float eps,
    const float weight_decay, const int step, const float lr, const float gnorm_scale, const int n
);

template <typename T, int OPTIMIZER>
__global__ void kOptimizer32bit2State(
    T* g, T* p, float* state1, float* state2, float* unorm, const float max_unorm, const float param_norm,
    const float beta1, const float beta2, const float beta3, const float alpha, const float eps,
    const float weight_decay, const int step, const float lr, const float gnorm_scale, const bool skip_zeros,
    const int n
);

template <typename T, int OPTIMIZER, int BLOCK_SIZE, int NUM_VALS>
__global__ void kPreconditionOptimizer32bit1State(
    T* g, T* p, float* state1, float* unorm, const float beta1, const float beta2, const float eps,
    const float weight_decay, const int step, const float lr, const float gnorm_scale, const int n
);

template <typename T, int OPTIMIZER>
__global__ void kOptimizer32bit1State(
    T* g, T* p, float* state1, float* unorm, const float max_unorm, const float param_norm, const float beta1,
    const float beta2, const float eps, const float weight_decay, const int step, const float lr,
    const float gnorm_scale, const bool skip_zeros, const int n
);

template <typename T, int OPTIMIZER, int BLOCK_SIZE, int N_PER_TH>
__global__ void kOptimizerStatic8bit2StateBlockwise(
    T* p, T* __restrict__ const g, unsigned char* state1, unsigned char* state2, const float beta1, const float beta2,
    const float beta3, const float alpha, const float eps, const int step, const float lr,
    float* __restrict__ const quantiles1, float* __restrict__ const quantiles2, float* absmax1, float* absmax2,
    float weight_decay, const float gnorm_scale, const bool skip_zeros, const int n
);

template <typename T, int OPTIMIZER, int BLOCK_SIZE, int N_PER_TH>
__global__ void kOptimizerStatic8bit1StateBlockwise(
    T* p, T* __restrict__ const g, unsigned char* state1, const float beta1, const float beta2, const float eps,
    const int step, const float lr, float* __restrict__ const quantiles1, float* absmax1, float weight_decay,
    const float gnorm_scale, const bool skip_zeros, const int n
);

template <typename T, int OPTIMIZER_BITS, int NUM_THREADS>
__global__ void kOptimizerStaticReadam1D(
    T* p, T* __restrict__ const g, unsigned char* state1, unsigned char* state2, const float beta1, const float beta2,
    const float eps, const int step, const float lr, float* __restrict__ const quantiles2, float* absmax2,
    float weight_decay, float decay_ratio, const float gnorm_scale, const bool skip_zeros, const int n
);

template <typename T, int OPTIMIZER_BITS, int NUM_THREADS>
__global__ void kOptimizerStaticReadam(
    T* p, T* __restrict__ const g, unsigned char* state1, float* __restrict__ const state2_row,
    float* __restrict__ const state2_col, float* __restrict__ const state2_mean, const float beta1, const float beta2,
    const float eps, const int step, const float lr, float weight_decay, float decay_ratio, const float gnorm_scale,
    const bool skip_zeros, const int n, const int n_rows, const int n_cols, const int stride_batch
);

template <typename T, int OPTIMIZER_BITS, int NUM_THREADS, int N_ROWS_PER_TILE>
__global__ void kOptimizerStaticReadamFused(
    T* p, T* __restrict__ const g, unsigned char* state1, float* __restrict__ const state2_row,
    float* __restrict__ const state2_col, float* __restrict__ const state2_mean, float* state2_accum_row,
    float* state2_accum_col, const float beta1, const float beta2, const float eps, const int step, const float lr,
    float weight_decay, float decay_ratio, const float gnorm_scale, const bool skip_zeros, const int n,
    const int n_batch, const int n_rows_per_batch, const int n_cols_per_batch, const int n_tileblocks_per_batch,
    const int n_rows_per_tileblock, const int stride_batch, const int stride_tile_per_tileblock, const int k_tiles
);

template <int NUM_THREADS>
__global__ void kReadamFinalizeStats(
    float* __restrict__ const state2_stat1, float* __restrict__ const state2_stat2,
    float* __restrict__ const state2_mean, float* __restrict__ const state2_accum_stat1,
    float* __restrict__ const state2_accum_stat2, const float beta2, const int n_batch, const int n_stat1,
    const int n_stat2
);

template <int ITEMS_PER_THREAD, int THREADS>
__global__ void kdequant_mm_int32_fp16(
    int* __restrict__ const A, float* __restrict__ const rowStats, float* __restrict__ const colStats, half* out,
    half* __restrict__ const bias, const int numRows, const int numCols, const int n
);

template <typename T, int THREADS, int SPARSE_DECOMP>
__global__ void kInt8VectorQuant(T* __restrict__ A, int8_t* out, float* rowStats, float threshold, int rows, int cols);

template <typename T, int THREADS, int BITS>
__global__ void kgemm_4bit_inference_naive(
    int M, int N, int K, T* __restrict__ const A, unsigned char* B, float* absmax, const float* datatype, T* out,
    int lda, int ldb, int ldc, int blocksize
);

template <typename T, int FUNC> __global__ void kfunc(T* A, T* B, T value, long n);

#endif
