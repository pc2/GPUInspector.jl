# Example: Data Bandwidth

!!! note
    **Test system:** DGX with 8x A100 GPUs (+ NVSwitch)

## Memory Bandwidth

You can benchmark the memory bandwidth of a GPU, i.e. how much data can be loaded within a second, with the function [`memory_bandwidth`](@ref). Under the hood, this function transfers a certain amount of data (`memcpy`), times the operation, and computes the resulting memory bandwidth.

```julia
julia> memory_bandwidth();
Memory Bandwidth (GiB/s):
 └ max: 1219.18
```

Note that this is reasonably close to the expected theoretical maximal memory bandwidth of our A100 test device.

```julia
julia> theoretical_memory_bandwidth();
Theoretical Maximal Memory Bandwidth (GiB/s):
 └ max: 1448.4
```

The scaling (as a function of data size) can be assessed with [`memory_bandwidth_scaling`](@ref).

```julia
julia> memory_bandwidth_scaling()

              ⠀⠀⠀⠀Peak: 1225.3 GiB/s (size = 1.0 GiB)⠀⠀⠀ 
              ┌────────────────────────────────────────┐ 
         2000 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
              │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
              │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
              │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
              │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
              │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀│ 
              │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⠔⠒⠉⠉⢸│ 
   GiB/s      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠒⠉⠀⠀⠀⠀⠀⠀⢸│ 
              │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡎⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
              │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
              │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
              │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
              │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
              │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠜⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
            0 │⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⡠⠔⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
              └────────────────────────────────────────┘ 
              ⠀2⁰⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2³⁰⠀ 
              ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀data size⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
```

### SAXPY
Note that we also provide similar functions that are based on a [SAXPY](https://developer.nvidia.com/blog/six-ways-saxpy/#:~:text=SAXPY%20stands%20for%20%E2%80%9CSingle%2DPrecision,and%20a%20scalar%20value%20A.) streaming kernel.

```julia
julia> memory_bandwidth_saxpy();
Memory Bandwidth (GiB/s):
 └ max: 1192.09

julia> memory_bandwidth_saxpy_scaling()

              ⠀⠀Peak: 1275.34 GiB/s (size = 300.0 MiB)⠀⠀ 
              ┌────────────────────────────────────────┐ 
         1280 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⢀│ 
              │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⠤⠤⠔⠒⠙⠊⠊⠉⢹│ 
              │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⠤⠔⠒⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
              │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠒⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
              │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠔⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
              │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠔⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
              │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
   GiB/s      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
              │⠀⠀⠀⠀⠀⠀⠀⡰⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
              │⠀⠀⠀⠀⠀⠀⡜⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
              │⠀⠀⠀⠀⢀⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
              │⠀⠀⠀⡠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
              │⠀⠀⡰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
              │⢀⠜⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
         1190 │⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ 
              └────────────────────────────────────────┘ 
              ⠀2²³⸱³²¹⁹⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2²⁸⸱²²⁸⁸⠀ 
              ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀vector length⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
```

## Host-to-Device Bandwidth

How much data can be transferred per second from the host memory to the GPU (and the other way around)?

```julia
julia> host2device_bandwidth()
Host <-> Device Bandwidth (GiB/s):
 └ max: 11.79

Host (pinned) <-> Device Bandwidth (GiB/s):
 └ max: 24.33
```

## Peer-to-Peer Bandwidth

### Unidirectional

```julia
julia> p2p_bandwidth();
Memsize: 38.147 MiB

Bandwidth (GiB/s):
 ├ max: 247.32
 ├ min: 173.5
 ├ avg: 229.63
 └ std_dev: 31.67

julia> p2p_bandwidth_all()
8×8 Matrix{Union{Nothing, Float64}}:
    nothing  245.706     241.075     244.467     246.434     242.229     245.085     245.033
 239.046        nothing  241.776     243.853     241.626     245.136     244.467     240.379
 246.957     242.633        nothing  242.937     245.291     248.114     239.193     242.684
 244.724     241.375     244.211        nothing  245.861     238.117     245.085     242.28
 241.576     246.329     242.582     245.602        nothing  246.59      240.677     243.343
 247.114     240.18      245.965     244.006     236.616        nothing  242.28      244.673
 243.802     242.028     248.326     239.933     244.365     245.033        nothing  245.498
 245.136     246.904     239.488     243.343     244.057     240.627     243.445        nothing
```

According to NVIDIA, the theoretical maximal peer-to-peer bandwidth for our A100 with NVSwitch is `300GB/s ≈ 279GiB/s`, which is in reasonable agreement with our findings.

### Bidirectional

```julia
julia> p2p_bandwidth_bidirectional();
Memsize: 38.147 MiB

Bandwidth (GiB/s):
 ├ max: 450.36
 ├ min: 448.66
 ├ avg: 449.69
 └ std_dev: 0.49

julia> p2p_bandwidth_bidirectional_all()
8×8 Matrix{Union{Nothing, Float64}}:
    nothing  456.631     453.133     454.946     453.67      453.953     455.06      454.662
 453.67         nothing  454.01      450.329     455.345     453.02      454.691     455.203
 453.981     451.53         nothing  452.344     453.868     454.747     452.232     454.208
 453.557     451.979     449.883        nothing  454.01      455.288     450.189     454.691
 452.429     454.293     445.094     454.151        nothing  453.472     451.474     453.981
 454.89      454.066     453.84      453.84      451.194        nothing  453.274     451.53
 453.925     453.02      454.293     456.459     451.839     451.951        nothing  455.032
 454.208     416.936     454.265     435.947     452.035     437.836     451.895        nothing
```