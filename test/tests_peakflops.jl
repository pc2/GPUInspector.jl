if backend() == NVIDIABackend()
    @testset "peakflops_gpu (CUDA cores)" begin
        @test typeof(peakflops_gpu(; verbose=false, tensorcores=false)) == Float64
        @test typeof(peakflops_gpu(; dtype=Float32, verbose=false, tensorcores=false)) ==
            Float64
        @test typeof(peakflops_gpu(; dtype=Float64, verbose=false, tensorcores=false)) ==
            Float64
    end

    @testset "peakflops_gpu (Tensor cores)" begin
        @test typeof(peakflops_gpu(; verbose=false, tensorcores=true)) == Float64
        @test typeof(peakflops_gpu(; dtype=Float16, verbose=false, tensorcores=true)) ==
            Float64
    end

    @testset "peakflops_gpu_matmul / scaling" begin
        @test typeof(CUDAExt.peakflops_gpu_matmul(; verbose=false)) == Float64
        @test typeof(
            CUDAExt.peakflops_gpu_matmul(; size=1024, dtype=Float64, verbose=false)
        ) == Float64
        @test typeof(CUDAExt.peakflops_gpu_matmul(; nmatmuls=2, nbench=2, verbose=false)) ==
            Float64
        @test typeof(CUDAExt.peakflops_gpu_matmul_scaling(; verbose=false)) ==
            Tuple{Vector{Int64},Vector{Float64}}
    end
elseif backend() == AMDBackend()
    # TODO
end
