@testitem "p2p_bandwidth" begin
    using LinearAlgebra
    using CUDA

    @testset "unidirectional" begin
        # p2p_bandwidth
        @test typeof(p2p_bandwidth(; verbose=false)) == Float64
        @test 0 ≤ p2p_bandwidth(; verbose=false)
        # options
        @test typeof(p2p_bandwidth(; memsize=MB(100), verbose=false)) == Float64
        @test typeof(p2p_bandwidth(; src=CuDevice(0), dst=CuDevice(1), verbose=false)) ==
            Float64
        @test typeof(p2p_bandwidth(; dtype=Float16, verbose=false)) == Float64
        @test typeof(p2p_bandwidth(; nbench=10, verbose=false)) == Float64
        @test typeof(p2p_bandwidth(; hist=true, verbose=true)) == Float64
        # p2p_bandwidth_all
        @test typeof(p2p_bandwidth_all(; verbose=false)) == Matrix{Union{Nothing,Float64}}
        Mp2p = p2p_bandwidth_all(; verbose=false)
        @test all(isnothing, diag(Mp2p))
        @test all(
            !isnothing(Mp2p[i, j]) for i in axes(Mp2p, 1), j in axes(Mp2p, 2) if i != j
        )
    end
    @testset "bidirectional" begin
        # p2p_bandwidth_bidirectional
        @test typeof(p2p_bandwidth_bidirectional(; verbose=false)) == Float64
        @test 0 ≤ p2p_bandwidth_bidirectional(; verbose=false)
        # options
        @test typeof(p2p_bandwidth_bidirectional(; memsize=MB(100), verbose=false)) == Float64
        @test typeof(p2p_bandwidth_bidirectional(; dtype=Float16, verbose=false)) == Float64
        @test typeof(p2p_bandwidth_bidirectional(; nbench=10, verbose=false)) == Float64
        @test typeof(p2p_bandwidth_bidirectional(; hist=true, verbose=true)) == Float64
        # p2p_bandwidth_bidirectional_all
        @test typeof(p2p_bandwidth_bidirectional_all(; verbose=false)) ==
            Matrix{Union{Nothing,Float64}}
        Mp2p = p2p_bandwidth_bidirectional_all(; verbose=false)
        @test all(isnothing, diag(Mp2p))
        @test all(
            !isnothing(Mp2p[i, j]) for i in axes(Mp2p, 1), j in axes(Mp2p, 2) if i != j
        )
    end
end

@testitem "host2device_bandwidth" begin
    using CUDA
    @test isnothing(host2device_bandwidth())
    @test isnothing(host2device_bandwidth(; memsize=MB(100)))
    @test isnothing(host2device_bandwidth(; dtype=Float16))
end

@testitem "memory_bandwidth" begin
    using CUDA
    @test typeof(memory_bandwidth()) == Float64
    @test typeof(memory_bandwidth(; memsize=MiB(10))) == Float64
    @test typeof(memory_bandwidth(; dtype=Float32)) == Float64

    @test typeof(memory_bandwidth_saxpy()) == Float64
    @test typeof(memory_bandwidth_saxpy(; size=2^20 * 2)) == Float64
    @test typeof(memory_bandwidth_saxpy(; dtype=Float32)) == Float64
end
