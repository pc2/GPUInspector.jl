@testset "gpuinfo / gpus" begin
    @test isnothing(gpus())
    @test isnothing(gpuinfo())
    @test isnothing(gpuinfo(GPUInspector.device()))
    if ngpus() > 1
        @test isnothing(gpuinfo_p2p_access())
    end
end
