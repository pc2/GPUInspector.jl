@testitem "gpuinfo / gpus" begin
    using CUDA
    @test isnothing(gpus())
    @test isnothing(gpuinfo())
    @test isnothing(gpuinfo(0))
    @test isnothing(gpuinfo(device()))
end
