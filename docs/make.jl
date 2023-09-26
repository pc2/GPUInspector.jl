using Documenter
using GPUInspector
using CUDA
using LinearAlgebra

BLAS.set_num_threads(1)

const src = "https://github.com/pc2/GPUInspector.jl"
const ci = get(ENV, "CI", "") == "true"

@info "Generating Documenter.jl site"
makedocs(;
    sitename="GPUInspector.jl",
    authors="Carsten Bauer",
    modules=[GPUInspector, Base.get_extension(GPUInspector, :CUDAExt)],
    checkdocs=:exports,
    doctest=false,
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
            "Backends" => "refs/backends.md",
            "CUDA Extension" => "refs/cuda_extension.md",
            "Utility" => "refs/utility.md",
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
    repo="$src/blob/{commit}{path}#{line}",
    format=Documenter.HTML(repolink="$src"; collapselevel=1, assets=["assets/favicon.ico"]),
)

if ci
    @info "Deploying documentation to GitHub"
    deploydocs(;
        repo="github.com/pc2/GPUInspector.jl",
        push_preview=true,
    )
end
