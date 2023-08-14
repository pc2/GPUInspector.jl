function monitoring_start(
    ::CUDABackend; freq=1, devices=CUDA.devices(), thread=Threads.nthreads(), verbose=true
)
    if ismonitoring()
        error("We are already monitoring.")
    end
    if thread <= 0 || thread > Threads.nthreads()
        throw(
            ArgumentError(
                "Invalid value for `thread` (must be `1 ≤ thread ≤ Threads.nthreads()`)."
            ),
        )
    end
    if Threads.nthreads() < length(devices) + 1
        @warn(
            "Notice that there aren't enough Julia threads to test all available GPUs and monitoring them at the same time."
        )
    end
    if eltype(devices) == CuDevice
        devs = _cudevice2nvmldevice.(devices)
    else
        devs = devices
    end
    verbose && @info("Spawning monitoring on Julia thread $thread.")
    # check which quantities we can monitor
    func_symbol_vec = Tuple{Function,Symbol}[]
    if all(supports_get_temperature(dev) for dev in devs)
        push!(func_symbol_vec, (get_temperatures, :temperature))
    end
    if all(supports_get_power_usage(dev) for dev in devs)
        push!(func_symbol_vec, (get_power_usages, :power))
    end
    if all(supports_get_gpu_utilization(dev) for dev in devs)
        push!(
            func_symbol_vec,
            (devs -> getproperty.(get_gpu_utilizations(devs), :compute), :compute),
        )
        push!(
            func_symbol_vec, (devs -> getproperty.(get_gpu_utilizations(devs), :mem), :mem)
        )
    end

    # spawn monitoring
    _monitoring!(true)
    t = @tspawnat thread _monitor(func_symbol_vec; freq, devices=devs)
    _set_monitoring_task(t)
    return nothing
end

"""
    monitoring_stop(; verbose=true) -> results

Specifically, `results` is a named tuple with the following keys:
* `time`: the (relative) times at which we measured
* `temperature`, `power`, `compute`, `mem`

(This method is from the CUDA backend.)
"""
function monitoring_stop(::CUDABackend; verbose=true)::MonitoringResults
    if ismonitoring()
        verbose && @info("Stopping monitoring and fetching results...")
        _monitoring!(false)
        results = fetch(_get_monitoring_task())
        return results
    else
        error("We aren't monitoring, so can't stop it...")
    end
end

function livemonitor_temperature(::CUDABackend, duration; kwargs...)
    return livemonitor_something(
        CUDABackend(),
        get_temperatures,
        duration;
        title="GPU Temperatures",
        ylabel="T [C]",
        kwargs...,
    )
end

function livemonitor_powerusage(::CUDABackend, duration; kwargs...)
    return livemonitor_something(
        CUDABackend(),
        get_power_usages,
        duration;
        title="GPU Power Usage",
        ylabel="P [W]",
        kwargs...,
    )
end

function livemonitor_something(
    ::CUDABackend,
    f::F,
    duration;
    freq=1,
    devices=NVML.devices(),
    plot=false,
    liveplot=true,
    ylims=nothing,
    title="",
    ylabel="Values",
) where {F}
    t_freq = 1.0 / freq
    values = Vector{Float64}[f(devices)]
    times = Float64[0.0]

    t = time()
    while (time() - t) < duration
        sleep(t_freq)
        push!(times, time() - t)
        push!(values, f(devices))
        if liveplot
            print("\033c")
            display(
                _plot_something(
                    values,
                    times;
                    devices,
                    ylims=isnothing(ylims) ? _defaultylims(values) : ylims,
                    title,
                    ylabel,
                ),
            )
        end
    end

    plot && display(
        _plot_something(
            values,
            times;
            devices,
            ylims=isnothing(ylims) ? _defaultylims(values) : ylims,
            title,
            ylabel,
        ),
    )
    return times, values
end

function _plot_something(
    values,
    times=1:length(values);
    devices=nothing,
    ylims=_defaultylims(values),
    title="",
    ylabel="Values",
)
    if isnothing(devices)
        device_labels = ["" for _ in 0:(length(first(values)) - 1)]
    else
        device_labels = [_device2string(dev) for dev in devices]
    end

    p = lineplot(
        times,
        getindex.(values, 1);
        title=title,
        xlabel="Time [s]",
        ylabel=ylabel,
        name=device_labels[1],
        ylim=ylims,
    )
    for i in 2:length(first(values))
        lineplot!(p, times, getindex.(values, i); name=device_labels[i])
    end
    return p
end

function _monitor(func_symbol_vec; freq, devices)
    nfuncs = length(func_symbol_vec)
    t_freq = 1.0 / freq
    # init result dict
    d = Dict{Symbol,Vector{Vector{Float64}}}()
    for i in 1:nfuncs
        f, s = func_symbol_vec[i]
        d[s] = Vector{Float64}[f(devices)]
    end
    times = [0.0]
    # monitor
    t = time()
    while ismonitoring()
        sleep(t_freq)
        push!(times, time() - t)
        for i in 1:nfuncs
            f, s = func_symbol_vec[i]
            push!(d[s], f(devices))
        end
    end
    device_info = Vector{Tuple{String,UUID}}(undef, length(devices))
    for (i, nvmldev) in enumerate(devices)
        dev = _nvmldevice2cudevice(nvmldev)
        device_info[i] = (_device2string(dev), uuid(dev))
    end
    return MonitoringResults(times, device_info, d)
end
