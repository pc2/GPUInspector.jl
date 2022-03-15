# Example: Peakflops

!!! note
    **Test system:** NVIDIA A100 GPU

## CUDA cores

We assess the peak performance of the CUDA cores by executing many pure [FMAs](https://de.wikipedia.org/wiki/Fused_multiply-add) (fused multiply-adds) on the CUDA cores in parallel.

Theoretically, we expect the performances below, which we compute from the clock rate, number of CUDA cores etc.
The same numbers are given by NVIDIA, e.g. [here](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) or [here](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf), or Wikipedia, e.g. [here](https://en.wikipedia.org/wiki/Ampere_(microarchitecture)#A100_accelerator_and_DGX_A100).

```julia
julia> theoretical_peakflops_gpu(; dtype=Float32, tensorcores=false);
Theoretical Peakflops (TFLOP/s):
 ├ tensorcores: false
 ├ dtype: Float32
 └ max: 19.5

julia> theoretical_peakflops_gpu(; dtype=Float64, tensorcores=false);
Theoretical Peakflops (TFLOP/s):
 ├ tensorcores: false
 ├ dtype: Float64
 └ max: 9.7
```

In good agreement, we find the following empirical numbers.

```julia
julia> peakflops_gpu(; dtype=Float32, tensorcores=false);
Peakflops (TFLOP/s):
 ├ tensorcores: false
 ├ dtype: Float32
 └ max: 19.1

julia> peakflops_gpu(; dtype=Float64, tensorcores=false);
Peakflops (TFLOP/s):
 ├ tensorcores: false
 ├ dtype: Float64
 └ max: 9.6

julia> peakflops_gpu(; dtype=Float16, tensorcores=false);
Peakflops (TFLOP/s):
 ├ tensorcores: false
 ├ dtype: Float16
 └ max: 12.8
```

## Tensor Cores

We assess the peak performance of the Tensor cores by executing many pure WMMAs (warp-level matrix-multiply-and-accumulate), see, e.g., [here](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/) and [here](https://developer.nvidia.com/blog/using-tensor-cores-in-cuda-fortran/)), on the Tensor cores in parallel.

Theoretically, we expect the performances below, which we compute from the clock rate, number of Tensor cores etc.
The same numbers are given by NVIDIA, e.g. [here](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) or [here](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf), or on Wikipedia, e.g. [here](https://en.wikipedia.org/wiki/Ampere_(microarchitecture)#A100_accelerator_and_DGX_A100).

```julia
julia> theoretical_peakflops_gpu(; dtype=Int8, tensorcores=true);
Theoretical Peakflops (TOP/s):
 ├ tensorcores: true
 ├ dtype: Int8
 └ max: 623.7

julia> theoretical_peakflops_gpu(; dtype=Float16, tensorcores=true);
Theoretical Peakflops (TFLOP/s):
 ├ tensorcores: true
 ├ dtype: Float16
 └ max: 311.9

julia> theoretical_peakflops_gpu(; dtype=Float32, tensorcores=true);
Theoretical Peakflops (TFLOP/s):
 ├ tensorcores: true
 ├ dtype: Float32
 └ max: 155.9
```

Empirically, we find the following numbers in good agreement.

```julia
julia> peakflops_gpu(; dtype=Int8, tensorcores=true); # as of writing, only works with CUDA.jl#master
Peakflops (TOP/s):
 ├ tensorcores: true
 ├ dtype: Int8
 └ max: 620.1

julia> peakflops_gpu(; dtype=Float16, tensorcores=true);
Peakflops (TFLOP/s):
 ├ tensorcores: true
 ├ dtype: Float16
 └ max: 311.2

julia> peakflops_gpu(; dtype=:TensorFloat32, tensorcores=true); # as of writing, requires Julia 1.8 and https://github.com/JuliaGPU/CUDA.jl/pull/1419
Peakflops (TFLOP/s):
 ├ tensorcores: true
 ├ dtype: TensorFloat32
 └ max: 155.5
```