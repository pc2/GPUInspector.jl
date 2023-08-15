@testitem "CUDA backend" begin
    using CUDA
    @test GPUInspector.is_cuda_loaded()
    @test GPUInspector.is_backend_loaded(NVIDIABackend())
    @test backend() == NVIDIABackend()
    @test isnothing(backend!(NoBackend()))
    @test backend() == NoBackend()
    @test isnothing(backend!(:cuda))
    @test backend() == NVIDIABackend()
end
