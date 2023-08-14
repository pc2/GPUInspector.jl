# GPUInspector.jl

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://pc2.github.io/GPUInspector.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://pc2.github.io/GPUInspector.jl/stable

[ci-img]: https://git.uni-paderborn.de/pc2-ci/julia/GPUInspector-jl/badges/main/pipeline.svg?key_text=CI@PC2
[ci-url]: https://git.uni-paderborn.de/pc2-ci/julia/GPUInspector-jl/-/pipelines

[cov-img]: https://codecov.io/gh/pc2/GPUInspector.jl/branch/main/graph/badge.svg
[cov-url]: https://codecov.io/gh/pc2/GPUInspector.jl

[lifecycle-img]: https://img.shields.io/badge/lifecycle-maturing-orange.svg

[code-style-img]: https://img.shields.io/badge/code%20style-blue-4495d1.svg
[code-style-url]: https://github.com/invenia/BlueStyle

<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
-->

*Inspecting high-performance (multi-)GPU systems with Julia*

| **Documentation**                                                               | **Build Status**                                                                                |  **Quality**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][ci-img]][ci-url] [![][cov-img]][cov-url] | ![][lifecycle-img] [![][code-style-img]][code-style-url] |

## Installation

The package is registered in the General registry and can readily be added by using the Pkg REPL mode.

```
] add GPUInspector
```

## Example

The package allows you to do various tests and benchmarks. Below we show a demonstrate a little stress test which lets a few A100 GPUs "burn" (i.e. lets them perform computations at close to peak performance) and monitors a few key metrics, such as power usage, temperature, and utilization, at the same time.

```julia
julia> using GPUInspector

julia> using CUDA # loading a GPU backend

julia> monitoring_start()                                                           
[ Info: Spawning monitoring on Julia thread 20.

julia> stresstest(; devices=CUDA.devices(), duration=10) # all devices, 10 seconds
[ Info: Will try to run for approximately 10 seconds on each GPU.
[ Info: Running StressTest{Float32} on Julia thread 4 and CuDevice(2).
[ Info: Running StressTest{Float32} on Julia thread 2 and CuDevice(0).
[ Info: Running StressTest{Float32} on Julia thread 6 and CuDevice(4).
[ Info: Running StressTest{Float32} on Julia thread 3 and CuDevice(1).
[ Info: Running StressTest{Float32} on Julia thread 9 and CuDevice(7).
[ Info: Running StressTest{Float32} on Julia thread 7 and CuDevice(5).
[ Info: Running StressTest{Float32} on Julia thread 5 and CuDevice(3).
[ Info: Running StressTest{Float32} on Julia thread 8 and CuDevice(6).
[ Info: Ran 11215 iterations on CuDevice(2).
[ Info: Ran 11241 iterations on CuDevice(6).
[ Info: Ran 11261 iterations on CuDevice(1).
[ Info: Ran 11236 iterations on CuDevice(5).
[ Info: Ran 11263 iterations on CuDevice(4).
[ Info: Ran 11261 iterations on CuDevice(3).
[ Info: Ran 11270 iterations on CuDevice(7).
[ Info: Ran 11241 iterations on CuDevice(0).
[ Info: Clearing GPU memory.
[ Info: Took 10.0 seconds to run the tests.

julia> results = monitoring_stop();
[ Info: Stopping monitoring and fetching results...

julia> plot_monitoring_results(results)

             ⠀⠀⠀⠀⠀⠀⠀⠀⠀GPU Utilization (Compute)
             ┌────────────────────────────────────────┐        
         105 │⠀⠀⠀⠀⡤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 0: NVIDIA A100-SXM4-40GB
             │⠀⠀⠀⢰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 1: NVIDIA A100-SXM4-40GB
             │⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 2: NVIDIA A100-SXM4-40GB
             │⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 3: NVIDIA A100-SXM4-40GB
             │⠀⠀⠀⡸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 4: NVIDIA A100-SXM4-40GB
             │⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 5: NVIDIA A100-SXM4-40GB
             │⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 6: NVIDIA A100-SXM4-40GB
   U [%]     │⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 7: NVIDIA A100-SXM4-40GB
             │⠀⠀⢰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                             
             │⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                             
             │⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                             
             │⠀⠀⡸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                             
             │⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                             
             │⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                             
           0 │⣀⣀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                             
             └────────────────────────────────────────┘                             
             ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀20⠀                             
             ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Time [s]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀                                                          
                                                                                                                 
            ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀GPU Temperature⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀                             
            ┌────────────────────────────────────────┐                             
         63 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 0: NVIDIA A100-SXM4-40GB
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠤⠤⠤⠒⢒⣲⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 1: NVIDIA A100-SXM4-40GB
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠤⠒⠊⣁⠤⣤⠤⠶⠮⠛⠛⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 2: NVIDIA A100-SXM4-40GB
            │⠀⠀⠀⠀⠀⠀⠀⡠⣒⣉⣉⣭⣓⠭⠛⠋⠉⠉⠉⠀⠀⠀⠀⠀⠀⢻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 3: NVIDIA A100-SXM4-40GB 
            │⠀⠀⠀⠀⠀⢠⣮⠮⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⢤⣤⠸⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 4: NVIDIA A100-SXM4-40GB 
            │⠀⠀⠀⠀⢠⡿⠁⠀⠀⠀⠀⠀⢀⡠⠤⣤⠤⠶⠮⠛⠋⣉⠭⠥⠤⡇⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 5: NVIDIA A100-SXM4-40GB 
            │⠀⠀⠀⢠⣷⠁⠀⣠⣒⠶⠒⢉⣉⢭⣛⣒⣒⡪⠝⠛⠛⠊⠉⠉⠉⣿⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 6: NVIDIA A100-SXM4-40GB 
   T [C]    │⠀⠀⠀⣼⠃⢠⠊⣁⠤⠔⠚⠓⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⡞⠲⡤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 7: NVIDIA A100-SXM4-40GB 
            │⠀⠀⢠⡏⢠⣿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣧⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                              
            │⠀⠀⣼⢡⣷⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                              
            │⠤⠤⡟⣸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⡳⡢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                              
            │⠉⠛⢇⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠪⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                              
            │⠀⠀⣸⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                              
            │⠒⠒⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                              
         28 │⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                              
            └────────────────────────────────────────┘                              
            ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀20⠀                              
            ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Time [s]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀                              
                                                                                                                 
             ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀GPU Power Usage⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀                              
             ┌────────────────────────────────────────┐                             
         340 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 0: NVIDIA A100-SXM4-40GB
             │⠀⠀⠀⠀⣤⣠⣤⣤⣶⡶⠶⠶⠶⠾⠿⠿⣶⣿⣷⣶⣶⣶⣒⣒⣺⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 1: NVIDIA A100-SXM4-40GB
             │⠀⠀⠀⢰⡯⡭⡿⠟⠛⠛⠛⠛⠛⠛⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠹⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 2: NVIDIA A100-SXM4-40GB
             │⠀⠀⠀⢸⠋⡜⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 3: NVIDIA A100-SXM4-40GB
             │⠀⠀⠀⢸⡰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 4: NVIDIA A100-SXM4-40GB
             │⠀⠀⠀⣾⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 5: NVIDIA A100-SXM4-40GB
             │⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 6: NVIDIA A100-SXM4-40GB
   P [W]     │⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ GPU 7: NVIDIA A100-SXM4-40GB
             │⠀⠀⢠⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                              
             │⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                              
             │⠀⠀⢸⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                              
             │⠀⠀⣼⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                              
             │⠀⠀⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                              
             │⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣶⣶⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                              
          56 │⣶⣶⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                              
             └────────────────────────────────────────┘                              
             ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀20⠀                              
             ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Time [s]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀                              
```

Note that we can get even fancier and plot the monitoring results on-the-fly, i.e. while the stress test is running. See e.g. [GPUInspector.livemonitor_temperature](https://pc2.github.io/GPUInspector.jl/dev/refs/monitoring/#GPUInspector.livemonitor_temperature-Tuple{Any}).

## Documentation

For more information, please find the [documentation](https://pc2.github.io/GPUInspector.jl/dev) here.
