# Example: CUDA/GPU Information

!!! note
    **Test system:** DGX with 8x A100 GPUs

```julia
julia> gpus()
8 devices:
  0: NVIDIA A100-SXM4-40GB (sm_80, 39.583 GiB / 39.586 GiB available)
  1: NVIDIA A100-SXM4-40GB (sm_80, 39.583 GiB / 39.586 GiB available)
  2: NVIDIA A100-SXM4-40GB (sm_80, 39.583 GiB / 39.586 GiB available)
  3: NVIDIA A100-SXM4-40GB (sm_80, 39.583 GiB / 39.586 GiB available)
  4: NVIDIA A100-SXM4-40GB (sm_80, 39.583 GiB / 39.586 GiB available)
  5: NVIDIA A100-SXM4-40GB (sm_80, 39.583 GiB / 39.586 GiB available)
  6: NVIDIA A100-SXM4-40GB (sm_80, 39.583 GiB / 39.586 GiB available)
  7: NVIDIA A100-SXM4-40GB (sm_80, 39.583 GiB / 39.586 GiB available)
```

More specific information for each gpu device can be obtained with [`gpuinfo(deviceid::Integer)`](@ref).

```
julia> gpuinfo()
Device: NVIDIA A100-SXM4-40GB (CuDevice(0))
Total amount of global memory: 42.5 GB
Number of CUDA cores: 6912
Number of multiprocessors: 108 (64 CUDA cores each)
GPU max. clock rate: 1410 Mhz
Memory clock rate: 1215 Mhz
Memory bus width: 5120-bit
L2 cache size: 41.9 MB
Max. texture dimension sizes (1D): 131072
Max. texture dimension sizes (2D): 131072, 65536
Max. texture dimension sizes (3D): 16384, 16384, 16384
Max. layered 1D texture size: 32768 (2048 layers)
Max. layered 2D texture size: 32768, 32768 (2048 layers)
Total amount of constant memory: 65.5 kB
Total amount of shared memory per block: 49.2 kB
Total number of registers available per block: 65536
Warp size: 32
Max. number of threads per multiprocessor: 2048
Max. number of threads per block: 1024
Max. dimension size of a thread block (x,y,z): 1024, 1024, 64
Max. dimension size of a grid size (x,y,z): 2147483647, 65535, 65535
Texture alignment: 512.0 B
Maximum memory pitch: 2.1 GB
Concurrent copy and kernel execution: Yes with 3 copy engine(s)
Run time limit on kernels: No
Integrated GPU sharing host memory: No
Support host page-locked memory mapping: Yes
Concurrent kernel execution: Yes
Alignment requirement for surfaces: Yes
Device has ECC support: Yes
Device supports Unified Addressing (UVA): Yes
Device supports managed memory: Yes
Device supports compute preemption: Yes
Supports cooperative kernel launch: Yes
Supports multi-device co-op kernel launch: Yes
Device PCI domain ID / bus ID / device ID: 0 / 7 / 0
Compute mode: Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)
```

Peer-to-peer access information can be retrived via [`gpuinfo_p2p_access`](@ref).

```julia
julia> gpuinfo_p2p_access()
P2P Access Supported:
8×8 Matrix{Bool}:
 0  1  1  1  1  1  1  1
 1  0  1  1  1  1  1  1
 1  1  0  1  1  1  1  1
 1  1  1  0  1  1  1  1
 1  1  1  1  0  1  1  1
 1  1  1  1  1  0  1  1
 1  1  1  1  1  1  0  1
 1  1  1  1  1  1  1  0

P2P Atomic Supported:
8×8 Matrix{Bool}:
 0  1  1  1  1  1  1  1
 1  0  1  1  1  1  1  1
 1  1  0  1  1  1  1  1
 1  1  1  0  1  1  1  1
 1  1  1  1  0  1  1  1
 1  1  1  1  1  0  1  1
 1  1  1  1  1  1  0  1
 1  1  1  1  1  1  1  0
```

Turns out that using [`CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED`](https://nw.tsuda.ac.jp/lec/cuda/doc_v9_0/html/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1gg578d7cf687ce20f7e99468e8c14e22dec7e28aec0cd03c462a49d00d1b145f46) or [`cuDeviceCanAccessPeer`](https://nw.tsuda.ac.jp/lec/cuda/doc_v9_0/html/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g496bdaae1f632ebfb695b99d2c40f19e) to query p2p access support may lead to different results (see [this](https://stackoverflow.com/questions/40258476/whats-the-difference-between-cudevicecanaccesspeer-and-cudevicegetp2pattri) stackoverflow thread). In `gpuinfo_p2p_access()` we use both methods and, if the results were to be different, we print both matrices (not shown above).