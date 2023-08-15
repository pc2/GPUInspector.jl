# GPUInspector.jl

## Installation

The package is registered in the General registry and can readily be added by using the Pkg REPL mode.

```
] add GPUInspector
```

**Note:** The minimal required Julia version is 1.6 but we strongly recommend to use Julia >= 1.7. Some features might not be available in Julia 1.6!

## Getting Started

### Backends

GPUInspector itself only provides limited functionality. Most of its features come to live - through [package extensions](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions)) - when you load a GPU backend, like CUDA.jl. Hence, most of the time, you want to add one of these backends next to GPUInspector to your Julia environment and then run

```julia
using GPUInspector
using CUDA # loading a GPU backend triggers the pkg extension to load
```

Note that you can check the current backend with the [`backend()`](@ref) function (and set it manually via [`backend!`](@ref)).

### Examples
```@contents
Pages = map(file -> joinpath("examples", file), readdir("examples"))
Depth = 1
```