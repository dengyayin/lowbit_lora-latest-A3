#include <cmath>
#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/Atomic.cuh>


using bf16 = at::BFloat16;
using athalf = at::Half;



__device__ __forceinline__ float extract_exp(float x){
    int a = *(int*)&x;
    a = a & 0x7F800000;
    float res = *(float*)&a;
    return res;
}


template <int N>
constexpr int get_mask(){
    if constexpr (N==4){
        return 0x7FF80000;
    }
    if constexpr (N==3){
        return 0x7FF00000;
    }
    if constexpr (N==2){
        return 0x7FE00000;
    }
    if constexpr (N==1){
        return 0x7FC00000;
    }
    if constexpr (N==0){
        return 0x7F800000;
    }
}


template <int N>
__device__ __forceinline__ float round_mant(float x, float e){
    // There is another 
    // float add is a good approx, but sometimes will result in wrong results. E.g. 0xBC7FFFE2 + 0.5f
    int x_int = *(int*) &x;
    int x_int_sign = (x_int & 0x80000000) | 0x3F800000;
    int x_int_unsigned = x_int & 0x7FFFFFFF;

    float x_unsigned = *(float*) &x_int_unsigned;
    float x_sign = *(float*) &x_int_sign;

    float mant = x_unsigned / e;
    float mant2 = mant * (float)(1<<N);
    float mant3 = std::floor(mant2 + 0.5);
    // printf("x:%f  m: %f  m2: %f  m3: %f\n", x, mant, mant2, mant3);
    if (mant3 == (float)(1<<(N+1))){
        mant3 = mant3 - 1;
    }
    mant3 = mant3 / (float)(1<<N);
    float out = x_sign * e * mant3;
    return out;
}


template <int N>
__device__ void hifxgv12_quant_cuda_inner(float* x_shared, float* res_shared, const int& thread_idx){
    float lv3[64], lv2[16], lv1[8], lv0;
    for (int i=0; i<64; ++i){
        lv3[i] = x_shared[thread_idx * 64 + i];
    }
    for (int i=0; i<16; ++i){
        lv2[i] = extract_exp(lv3[i*4]);
        for (int j=1; j<4; ++j){
            float e = extract_exp(lv3[i*4+j]);
            if (e>lv2[i]){ lv2[i] = e; }
        }
    }
    for (int i=0; i<8; ++i){
        lv1[i] = (lv2[i*2]>lv2[i*2+1]) ? lv2[i*2] : lv2[i*2+1];
    }
    lv0 = lv1[0];
    for (int i=0; i<8; ++i){
        lv0 = (lv0 < lv1[i]) ? lv1[i] : lv0;
    }
    lv0 = lv0 / 4.0f;
    if (lv0 < 3.5527136e-15){
        for (int i=0; i<64; ++i){
            res_shared[thread_idx * 64 + i] = 0;
        }
    }else{
        for (int i=0; i<8; ++i){
            lv1[i] = lv1[i] / lv0 / 2.0f;
            lv1[i] = (lv1[i] < 1.0f) ? 1.0f : lv1[i];
        }
        for (int i=0; i<8; ++i){
            lv2[i*2] = lv2[i*2] / lv1[i] / lv0;
            lv2[i*2] = (lv2[i*2] < 1.0f) ? 1.0f : lv2[i*2];
            lv2[i*2] = lv0 * lv1[i] * lv2[i*2];
            lv2[i*2+1] = lv2[i*2+1] / lv1[i] / lv0;
            lv2[i*2+1] = (lv2[i*2+1] < 1.0f) ? 1.0f : lv2[i*2+1];
            lv2[i*2+1] = lv0 * lv1[i] * lv2[i*2+1];
        }
        
        for (int i=0; i<16; ++i){
            // printf("LV3  %d  :  %f  :  %f\n", i*4+0, lv3[i*4+0], lv2[i]);
            // printf("LV3  %d  :  %f  :  %f\n", i*4+1, lv3[i*4+1], lv2[i]);
            // printf("LV3  %d  :  %f  :  %f\n", i*4+2, lv3[i*4+2], lv2[i]);
            // printf("LV3  %d  :  %f  :  %f\n", i*4+3, lv3[i*4+3], lv2[i]);
            lv3[i*4] = round_mant<N>(lv3[i*4], lv2[i]);
            lv3[i*4+1] = round_mant<N>(lv3[i*4+1], lv2[i]);
            lv3[i*4+2] = round_mant<N>(lv3[i*4+2], lv2[i]);
            lv3[i*4+3] = round_mant<N>(lv3[i*4+3], lv2[i]);
        }
        for (int i=0; i<64; ++i){
            res_shared[thread_idx * 64 + i] = lv3[i];
        }
    }
    
}


template <int N>
__global__ void hifxgv12_quant_cuda_kernel(const float* x_ori, float* res_ori, const int n_total){
    constexpr int G=32;
    const int thread_idx = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int block_size = blockDim.x;

    int offset = block_idx *  2 * G * block_size;
    const float* x_offset = x_ori + offset;
    float* res_mem = res_ori + offset;

    __shared__ float x_shared[4096];  // max shared_size 
    __shared__ float res_shared[4096];

    // memory read 
    for (int i=0;i<2*G;++i){
        if (offset + block_size*i + thread_idx >= n_total){
            x_shared[block_size*i + thread_idx] = 0;
        }else{
            x_shared[block_size*i + thread_idx] = x_offset[block_size*i + thread_idx];
        }
    }
    __syncthreads();
    
    hifxgv12_quant_cuda_inner<N>(x_shared, res_shared, thread_idx);

    // write to memory (this will result in continuous write to memory, thus faster)
    __syncthreads();
    for (int i=0;i<2*G;++i){
        if (offset + block_size*i + thread_idx >= n_total){
            // res_mem[block_size*i + thread_idx] = 0;
        }else{
            res_mem[block_size*i + thread_idx] = res_shared[block_size*i + thread_idx];
        }
    }
}



void hifxgv12_quant_cuda(torch::Tensor x, torch::Tensor result, int N){
    int threads = 4096 / 2 / 32;
    int blocks = (x.numel() + 4095) / 4096;
    
    if ((N==2)){
        hifxgv12_quant_cuda_kernel<2><<<blocks, threads>>>(x.data_ptr<float>(), result.data_ptr<float>(), x.numel());
    }
    if ((N==3)){
        hifxgv12_quant_cuda_kernel<3><<<blocks, threads>>>(x.data_ptr<float>(), result.data_ptr<float>(), x.numel());
    }
    if ((N==1)){
        hifxgv12_quant_cuda_kernel<1><<<blocks, threads>>>(x.data_ptr<float>(), result.data_ptr<float>(), x.numel());
    }
    if ((N==0)){
        hifxgv12_quant_cuda_kernel<0><<<blocks, threads>>>(x.data_ptr<float>(), result.data_ptr<float>(), x.numel());
    }

}


__global__ void hif8_quant_cuda_kernel(const float* x_ori, float* res_ori, const int n_total){
    const int thread_idx = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int block_size = blockDim.x;

    int offset = block_idx * block_size;
    const float* x_offset = x_ori + offset;
    float* res_mem = res_ori + offset;


    // memory read 
    if (offset + thread_idx < n_total){
        float x = x_offset[thread_idx];
        uint x_int = *(uint*)&x;
        uint exp_x = (x_int & 0x7F800000);
        
        int mant_bits = 0;
        if ((exp_x<=0x47000000) && (exp_x>=0x38000000)){
            mant_bits = 1;
        }
        if ((exp_x<=0x43000000) && (exp_x>=0x3C000000)){
            mant_bits = 2;
        }
        if ((exp_x<=0x41000000) && (exp_x>=0x3E000000)){
            mant_bits = 3;
        }

        // round mantissa 
        uint mant = (x_int & 0x007FFFFF) >> (22 - mant_bits);
        mant = ((mant + 1) >> 1);
        mant = mant << (23 - mant_bits);
        x_int = (x_int & 0xFF800000) + mant;
        float res = *(float*)&x_int;
        res_mem[thread_idx] = res;
    }
}

void hif8_quant_cuda(torch::Tensor x, torch::Tensor result){
    int threads = 1024;
    int blocks = (x.numel() + 1023) / 1024;
    hif8_quant_cuda_kernel<<<blocks, threads>>>(x.data_ptr<float>(), result.data_ptr<float>(), x.numel());
}
