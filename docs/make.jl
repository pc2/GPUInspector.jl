# push!(LOAD_PATH,"../src/")
using Documenter
using DocThemePC2
using GPUInspector
using LinearAlgebra
using MKL # optional

BLAS.set_num_threads(1)

const src = "https://git.uni-paderborn.de/pc2/julia/GPUInspector.jl"
const ci = get(ENV, "CI", "") == "true"

@info "Preparing DocThemePC2"
DocThemePC2.install(@__DIR__)

@info "Generating Documenter.jl site"
makedocs(;
    sitename="GPUInspector.jl",
    authors="Carsten Bauer",
    modules=[GPUInspector],
    checkdocs=:exports,
    pages=[
        "GPUInspector" => "index.md",
        "Examples" => [
            "GPU Information" => "examples/gpuinfo.md",
            "Data Bandwidth" => "examples/data_bandwidth.md",
            "Peakflops" => "examples/peakflops_gpu.md",
            "GPU Stress Test" => "examples/gpustresstest.md",
        ],
        "Explanations" => ["DGX Details" => "explanations/dgx.md"],
        "References" => [
            "GPU Information" => "refs/gpuinfo.md",
            "Data Bandwidth" => "refs/data_bandwidth.md",
            "Peakflops" => "refs/peakflops_gpu.md",
            "GPU Stress Test" => "refs/gpustresstest.md",
            "CPU Stress Test" => "refs/stresstest_cpu.md",
            "GPU Monitoring" => "refs/monitoring.md",
            "CUDA Wrappers" => "refs/cuda_wrappers.md",
            "Utility" => "refs/utility.md",
            "Worker Utilities" => "refs/workers.md",
            "HDF5" => "refs/hdf5.md",
        ],
        "Tested Devices" => [
            "A100 SXM2" => "devices/a100_sxm2.md",
            "V100 SXM2" => "devices/v100_sxm2.md",
            "GeForce GTX 1650" => "devices/geforce_gtx_1650.md",
        ],
        # "Development Docs" => [
        #     "Contribution Guide" => "devel/guide.md",
        #     "Benchmarking" => "devel/benchmarking.md",
        #     "Testing" => "devel/testing.md",
        #     "Analysis" => "devel/analysis.md",
        # ],
    ],
    # assets = ["assets/custom.css", "assets/custom.js"]
    repo="https://git.uni-paderborn.de/pc2/julia/GPUInspector.jl/blob/{commit}{path}#{line}",
    format=Documenter.HTML(; collapselevel=1, assets=["assets/favicon.ico"]),
)

if ci
    @info "Deploying documentation to GitHub"
    deploydocs(;
        repo="github.com/pc2/GPUInspector.jl",
        push_preview=true,
    )
end