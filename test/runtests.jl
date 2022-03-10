using GPUInspector
using Test
using CUDA
using InteractiveUtils: subtypes
using LinearAlgebra

if !GPUInspector.functional()
    error("Can't run testsuite since CUDA/GPU not present or not functional!")
end
if Threads.nthreads() == 1
    error("Can't run testsuite in single-threaded mode. Start Julia with multiple threads.")
end
if Threads.nthreads() < length(CUDA.devices()) + 1
    # we should have at least one thread per gpu + one monitoring thread
    @warn(
        "You should run the tests with at least $(length(CUDA.devices()) + 1) Julia threads.",
        Threads.nthreads(),
        length(CUDA.devices())
    )
end

@testset "GPUInspector.jl" begin
    @testset "UnitPrefixedBytes" begin
        # general stuff
        b = B(40_000_000)
        m = MiB(38.14697265625)
        m̃ = MB(40.0)
        @test typeof(b) == B
        @test typeof(m) == MiB
        @test typeof(m̃) == MB
        @test string(b) == "4.0e7 B"
        @test string(m) == "~38.15 MiB"
        @test string(m̃) == "40.0 MB"
        @test value(b) == 40_000_000
        @test value(m) ≈ 38.14697265625
        @test value(m̃) ≈ 40.0
        @test simplify(b) ≈ m
        @test simplify(m) ≈ m
        @test simplify(m̃) ≈ m
        @test simplify(b; base=10) ≈ m̃
        @test simplify(m; base=10) ≈ m̃
        @test simplify(m̃; base=10) ≈ m̃
        @test change_base(b) == b
        @test change_base(m) == m̃
        @test change_base(m̃) == m
        @test convert(MB, b) == m̃
        @test convert(MiB, b) == m
        @test convert(MB, m) == m̃
        @test convert(MiB, m̃) == m
        @test bytes(b) == 4.0e7
        @test bytes(m) == 4.0e7
        @test bytes(m̃) == 4.0e7
        @test bytes(4.0e7) ≈ m

        # GiB <-> GB conversion because it's particularly important
        @test convert(GiB, GB(300)) ≈ GiB(279.39677238464355)
        @test change_base(GB(300)) ≈ GiB(279.39677238464355)
        @test convert(GB, GiB(300)) ≈ GB(322.12254720000004)
        @test change_base(GiB(300)) ≈ GB(322.12254720000004)

        # ≈, ==, ===
        @test b == m
        @test b == m̃
        @test m == m̃
        @test b ≈ m
        @test b ≈ m̃
        @test m ≈ m̃
        @test b === b
        @test b !== m
        @test b !== m̃
        @test m !== m̃

        # basic arithmetics
        types = subtypes(UnitPrefixedBytes)
        for T in types
            @test T(1.2) + T(2.3) ≈ T(3.5)
            @test T(3.5) - T(2.3) ≈ T(1.2)
            @test 2 * T(1.234) ≈ T(2.468)
            @test T(10) / 2 == T(5)
        end
        for T in types, S in types
            @test bytes(T(1.2) + S(2.3)) ≈ bytes(T(1.2)) + bytes(S(2.3))
            @test abs(bytes(T(1.2) - S(2.3))) ≈ abs(bytes(T(1.2)) - bytes(S(2.3)))
        end
        @test B(40_000_000) + MB(3) - 2 * KiB(2) ≈ MB(42.995904)
    end

    @testset "toggle_tensorcoremath" begin
        @test isnothing(toggle_tensorcoremath(true; verbose=false))
        @test CUDA.math_mode() == CUDA.FAST_MATH
        @test isnothing(toggle_tensorcoremath(false; verbose=false))
        @test CUDA.math_mode() == CUDA.DEFAULT_MATH
        # test toggle
        @test isnothing(toggle_tensorcoremath(; verbose=false)) 
        @test CUDA.math_mode() == CUDA.FAST_MATH
        @test isnothing(toggle_tensorcoremath(; verbose=false)) 
        @test CUDA.math_mode() == CUDA.DEFAULT_MATH
    end

    @testset "gpu stresstest" begin
        @testset "different kinds" begin
            for dev in (device(), devices()) # single- and multi-gpu
                @test isnothing(stresstest(dev; duration=2, verbose=false))
                @test isnothing(stresstest(dev; enforced_duration=2, verbose=false))
                @test isnothing(stresstest(dev; approx_duration=2, verbose=false))
                @test isnothing(stresstest(dev; niter=100, verbose=false))
                @test isnothing(stresstest(dev; niter=100, verbose=false))
                @test isnothing(stresstest(dev; mem=0.2, verbose=false))
            end
        end

        @testset "keyword options" begin
            dev = device()
            @test isnothing(stresstest(dev; duration=2, verbose=false))
            @test isnothing(stresstest(dev; duration=2, size=3000, verbose=false))
            @test isnothing(stresstest(dev; duration=2, dtype=Float16, verbose=false))
            @test isnothing(stresstest(dev; duration=2, clearmem=true, verbose=false))
            @test isnothing(stresstest(dev; duration=2, batch_duration=1, verbose=false))
            # TODO: kwargs: threads, parallel
        end

        @testset "monitoring" begin
            @testset "automatic (monitoring=true)" begin
                for dev in (device(), devices()) # single- and multi-gpu
                    @test typeof(
                        stresstest(dev; duration=2, verbose=false, monitoring=true)
                    ) == MonitoringResults
                end
            end
            @testset "manual" begin
                for devs in ([device()], devices()) # single- and multi-gpu
                    @test isnothing(monitoring_start(; freq=1, devices=devs, verbose=false))
                    @test isnothing(
                        stresstest(devs; duration=2, verbose=false, monitoring=false)
                    )
                    @test typeof(monitoring_stop(; verbose=false)) == MonitoringResults
                end
                # thread kwarg
                # @test isnothing(monitoring_start(; freq=1, devices=[device()], verbose=false, thread=2))
                # sleep(0.5)
                # TODO: How to find the threadid() of a Task?
                # @test isnothing(monitoring_stop(; verbose=false))
            end
        end

        @testset "monitoring results" begin
            @testset "MonitoringResults" begin
                r = stresstest(device(); duration=2, verbose=false, monitoring=true)
                @test typeof(r) == MonitoringResults
                @test typeof(r.times) == Vector{Float64}
                @test typeof(r.devices) == Vector{Tuple{String,Base.UUID}}
                quant = first(keys(r.results))
                @test r.results[quant] == getproperty(r, quant)
            end
            @testset "save / load" begin
                d = Dict{Symbol,Vector{Vector{Float64}}}()
                ndevs = length(CUDA.devices())
                d[:asd] = [rand(ndevs) for _ in 1:5]
                d[:qwe] = [rand(ndevs) for _ in 1:5]
                d[:jkl] = [rand(ndevs) for _ in 1:5]
                devices = Tuple{String,Base.UUID}[
                    (GPUInspector._device2string(dev), uuid(dev)) for
                    dev in collect(CUDA.devices())
                ]
                r = MonitoringResults(rand(5), devices, d)
                cd(mktempdir()) do
                    save_monitoring_results("tmp.h5", r)
                    r2 = load_monitoring_results("tmp.h5")
                    @test r == r2
                end
            end
            @testset "plot / savefig" begin
                r = load_monitoring_results(joinpath(@__DIR__, "test.h5"))
                @testset "UnicodePlots (tofile=false)" begin
                    @test isnothing(plot_monitoring_results(r))
                    @test isnothing(plot_monitoring_results(r, (:compute, :mem)))
                end
                @testset "CairoMakie (tofile=true)" begin
                    cd(mktempdir()) do
                        @test isnothing(plot_monitoring_results(r; tofile=true))
                        @test isnothing(
                            plot_monitoring_results(r, (:compute, :mem); tofile=true)
                        )
                        @test isnothing(plot_monitoring_results(r; tofile=true, ext=:png))
                        @test isnothing(plot_monitoring_results(r; tofile=true, ext=:pdf))
                    end
                end
            end
        end
    end

    @testset "gpuinfo / gpus" begin
        @test isnothing(gpus())
        @test isnothing(gpuinfo())
        @test isnothing(gpuinfo(0))
        @test isnothing(gpuinfo(device()))
    end

    @testset "p2p_bandwidth" begin
        @testset "unidirectional" begin
            # p2p_bandwidth
            @test typeof(p2p_bandwidth(; verbose=false)) == Float64
            @test 0 ≤ p2p_bandwidth(; verbose=false)
            # options
            @test typeof(p2p_bandwidth(MB(100); verbose=false)) == Float64
            @test typeof(
                p2p_bandwidth(; src=CuDevice(0), dst=CuDevice(1), verbose=false)
            ) == Float64
            @test typeof(p2p_bandwidth(; dtype=Float16, verbose=false)) == Float64
            @test typeof(p2p_bandwidth(; nbench=10, verbose=false)) == Float64
            @test typeof(p2p_bandwidth(; hist=true, verbose=true)) == Float64
            # p2p_bandwidth_all
            @test typeof(p2p_bandwidth_all(; verbose=false)) ==
                Matrix{Union{Nothing,Float64}}
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
            @test typeof(p2p_bandwidth_bidirectional(MB(100); verbose=false)) == Float64
            @test typeof(p2p_bandwidth_bidirectional(; dtype=Float16, verbose=false)) ==
                Float64
            @test typeof(p2p_bandwidth_bidirectional(; nbench=10, verbose=false)) ==
                Float64
            @test typeof(p2p_bandwidth_bidirectional(; hist=true, verbose=true)) ==
                Float64
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

    @testset "host2device_bandwidth" begin
        @test isnothing(host2device_bandwidth())
        @test isnothing(host2device_bandwidth(MB(100)))
        @test isnothing(host2device_bandwidth(; dtype=Float16))
    end

    @testset "memory_bandwidth" begin
        @test typeof(memory_bandwidth()) == Float64
        @test typeof(memory_bandwidth(MiB(10))) == Float64
        @test typeof(memory_bandwidth(; dtype=Float32)) == Float64

        @test typeof(memory_bandwidth_saxpy()) == Float64
        @test typeof(memory_bandwidth_saxpy(; size=2^20*2)) == Float64
        @test typeof(memory_bandwidth_saxpy(; dtype=Float32)) == Float64
    end

    @testset "peakflops_gpu (CUDA cores)" begin
        @test typeof(peakflops_gpu(; verbose=false, tensorcores=false)) == Float64
        @test typeof(peakflops_gpu(; dtype=Float32, verbose=false, tensorcores=false)) == Float64
        @test typeof(peakflops_gpu(; dtype=Float64, verbose=false, tensorcores=false)) == Float64
    end

    @testset "peakflops_gpu (Tensor cores)" begin
        @test typeof(peakflops_gpu(; verbose=false, tensorcores=true)) == Float64
        @test typeof(peakflops_gpu(; dtype=Float16, verbose=false, tensorcores=true)) == Float64
    end

    @testset "peakflops_gpu_matmul / scaling" begin
        @test typeof(peakflops_gpu_matmul(; verbose=false)) == Float64
        @test typeof(peakflops_gpu_matmul(; size=1024, dtype=Float64, verbose=false)) == Float64
        @test typeof(peakflops_gpu_matmul(; nmatmuls=2, nbench=2, verbose=false)) == Float64
        @test typeof(peakflops_gpu_matmul_scaling(; verbose=false)) == Tuple{Vector{Int64}, Vector{Float64}}
    end
end