# GPUInspector.jl

## Installation

The package is **not** registered in the General registry, but only in the [PC2Registry](https://git.uni-paderborn.de/pc2/julia/PC2Registry).
If you have subscribed to the latter, you can simply add the package as follows.

```
] add GPUInspector.jl
```

Otherwise, you can readily add it by using the explicit URL.

```
] add https://github.com/pc2/GPUInspector.jl
```

**Note:** The minimal required Julia version is 1.7.

## Getting Started

```@contents
Pages = map(file -> joinpath("examples", file), readdir("examples"))
Depth = 1
```