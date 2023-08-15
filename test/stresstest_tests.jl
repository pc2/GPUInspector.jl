@testitem "Stresstest: different kinds" begin
    using CUDA
    @test isnothing(stresstest(; duration=2, verbose=false))
    @test isnothing(stresstest(; enforced_duration=2, verbose=false))
    @test isnothing(stresstest(; approx_duration=2, verbose=false))
    @test isnothing(stresstest(; niter=100, verbose=false))
    @test isnothing(stresstest(; niter=100, verbose=false))
    @test isnothing(stresstest(; mem=0.2, verbose=false))
end

@testitem "Stresstest: keyword options" begin
    using CUDA
    @test isnothing(stresstest(; duration=2, verbose=false))
    @test isnothing(stresstest(; duration=2, devices=devices(), verbose=false))
    @test isnothing(stresstest(; duration=2, size=3000, verbose=false))
    @test isnothing(stresstest(; duration=2, dtype=Float16, verbose=false))
    @test isnothing(stresstest(; duration=2, clearmem=true, verbose=false))
    @test isnothing(stresstest(; duration=2, batch_duration=1, verbose=false))
    # TODO: kwargs: threads, parallel
end

@testitem "Stresstest: monitoring" begin
    using CUDA
    @testset "automatic (monitoring=true)" begin
        @test typeof(
            stresstest(; devices=devices(), duration=2, verbose=false, monitoring=true)
        ) == MonitoringResults
    end
    @testset "manual" begin
        devs = devices()
        @test isnothing(monitoring_start(; freq=1, devices=devs, verbose=false))
        @test isnothing(
            stresstest(; devices=devs, duration=2, verbose=false, monitoring=false)
        )
        @test typeof(monitoring_stop(; verbose=false)) == MonitoringResults
        # thread kwarg
        # @test isnothing(monitoring_start(; freq=1, devices=[device()], verbose=false, thread=2))
        # sleep(0.5)
        # TODO: How to find the threadid() of a Task?
        # @test isnothing(monitoring_stop(; verbose=false))
    end
end

@testitem "Stresstest: monitoring results" begin
    using CUDA
    @testset "MonitoringResults" begin
        r = stresstest(; duration=2, verbose=false, monitoring=true)
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
            (CUDAExt._device2string(dev), uuid(dev)) for dev in collect(CUDA.devices())
        ]
        r = MonitoringResults(rand(5), devices, d)
        cd(mktempdir()) do
            save_monitoring_results("tmp.h5", r)
            r2 = load_monitoring_results("tmp.h5")
            @test r == r2
        end
    end
    @testset "plot" begin
        r = load_monitoring_results(joinpath(@__DIR__, "test.h5"))
        @test isnothing(plot_monitoring_results(r))
        @test isnothing(plot_monitoring_results(r, (:compute, :mem)))
    end
end

@testitem "Stresstest: monitoring results (CairoMakie)" begin
    using CairoMakie
    r = load_monitoring_results(joinpath(@__DIR__, "test.h5"))
    @test isnothing(savefig_monitoring_results(r))
    @test isnothing(savefig_monitoring_results(r, (:compute, :mem)))
    @test isnothing(savefig_monitoring_results(r; ext=:png))
    @test isnothing(savefig_monitoring_results(r; ext=:pdf))
    rm.(filter(endswith(".pdf"), readdir())) # cleanup
end
