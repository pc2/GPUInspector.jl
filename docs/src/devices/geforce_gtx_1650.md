# NVIDIA GeForce GTX 1650 (sm_75, 3.787 GiB)

Some information by NVIDIA is available [here](https://www.nvidia.com/de-de/geforce/graphics-cards/gtx-1650/). Some external numbers may be found [here](https://www.anandtech.com/show/14255/nvidia-launches-geforce-gtx-1650-budget-turing-for-149-available-today).

## Peakflops

### Theoretical

```julia
julia> theoretical_peakflops_gpu(; dtype=Float32);
Theoretical Peakflops (TFLOP/s):
 └ max: 2.9

julia> theoretical_peakflops_gpu(; dtype=Float64);
Theoretical Peakflops (TFLOP/s):
 └ max: 1.5
```

### Empirical

```julia
julia> peakflops_gpu_fmas(; dtype=Float32); # too large?
Peakflops (TFLOP/s):
 └ max: 3.41

julia> peakflops_gpu_fmas(; dtype=Float64); # too low?
Peakflops (TFLOP/s):
 └ max: 0.11
```

## Memory bandwidth

```julia
julia> memory_bandwidth();
Memory Bandwidth (GiB/s):
 └ max: 159.33

julia> GiB(159.33) |> change_base
~171.08 GB
```

```julia
julia> host2device_bandwidth()
Memsize: 512.000 MiB
GPU: CuDevice(0) - NVIDIA GeForce GTX 1650

Host <-> Device Bandwidth (GiB/s):
 └ max: 12.06

Host (pinned) <-> Device Bandwidth (GiB/s):
 └ max: 12.05

Device <-> Device (same device) Bandwidth (GiB/s):
 └ max: 153.0
```

## GPU information

```julia
julia> CUDA.versioninfo()
CUDA toolkit 11.6, artifact installation
NVIDIA driver 470.86.0, for CUDA 11.4
CUDA driver 11.4

Libraries: 
- CUBLAS: 11.8.1
- CURAND: 10.2.9
- CUFFT: 10.7.0
- CUSOLVER: 11.3.2
- CUSPARSE: 11.7.1
- CUPTI: 16.0.0
- NVML: 11.0.0+470.86
- CUDNN: 8.30.2 (for CUDA 11.5.0)
- CUTENSOR: 1.4.0 (for CUDA 11.5.0)

Toolchain:
- Julia: 1.7.2
- LLVM: 12.0.1
- PTX ISA support: 3.2, 4.0, 4.1, 4.2, 4.3, 5.0, 6.0, 6.1, 6.3, 6.4, 6.5, 7.0
- Device capability support: sm_35, sm_37, sm_50, sm_52, sm_53, sm_60, sm_61, sm_62, sm_70, sm_72, sm_75, sm_80

1 device:
  0: NVIDIA GeForce GTX 1650 (sm_75, 2.240 GiB / 3.787 GiB available)

julia> gpuinfo()
Device: NVIDIA GeForce GTX 1650 (CuDevice(0))
Total amount of global memory: 3.787 GiB
Number of CUDA cores: 896
Number of multiprocessors: 14 (64 CUDA cores each)
GPU max. clock rate: 1620 MHz
Memory clock rate: 6001 MHz
Memory bus width: 128-bit
L2 cache size: 1024.000 KiB
Max. texture dimension sizes (1D): 131072
Max. texture dimension sizes (2D): 131072, 65536
Max. texture dimension sizes (3D): 16384, 16384, 16384
Max. layered 1D texture size: 32768 (2048 layers)
Max. layered 2D texture size: 32768, 32768 (2048 layers)
Total amount of constant memory: 64.000 KiB
Total amount of shared memory per block: 48.000 KiB
Total number of registers available per block: 65536
Warp size: 32
Max. number of threads per multiprocessor: 1024
Max. number of threads per block: 1024 
Max. dimension size of a thread block (x,y,z): 1024, 1024, 64
Max. dimension size of a grid size (x,y,z): 2147483647, 65535, 65535
Texture alignment: 512 bytes
Maximum memory pitch: 2.000 GiB
Concurrent copy and kernel execution: Yes with 3 copy engine(s)
Run time limit on kernels: Yes
Integrated GPU sharing host memory: No 
Support host page-locked memory mapping: Yes
Concurrent kernel execution: Yes
Alignment requirement for surfaces: Yes
Device has ECC support: No
Device supports Unified Addressing (UVA): Yes
Device supports managed memory: Yes
Device supports compute preemption: Yes
Supports cooperative kernel launch: Yes
Supports multi-device co-op kernel launch: Yes
Device PCI domain ID / bus ID / device ID: 0 / 1 / 0
Compute mode: Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)
```