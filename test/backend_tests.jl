@testitem "CUDA backend" begin
    using CUDA
    @test GPUInspector.is_cuda_loaded()
    @test GPUInspector.is_backend_loaded(CUDABackend())
    @test backend() == CUDABackend()
    @test isnothing(backend!(NoBackend()))
    @test backend() == NoBackend()
    @test isnothing(backend!(:cuda))
    @test backend() == CUDABackend()
end
