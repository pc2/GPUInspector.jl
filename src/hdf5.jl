"""
    save_monitoring_results(filename::String, r::MonitoringResults; overwrite=false)
Store the given `MonitoringResults` (output of [`monitoring_stop`](@ref)) to disk as an HDF5 file with name `filename`.
"""
function save_monitoring_results(filename::String, r::MonitoringResults; overwrite=false)
    if isfile(filename) && !overwrite
        error(
            "Output file $filename already exists. If you want to overwrite, set `overwrite=true`.",
        )
    end

    h5open(filename, "w") do f
        f["times"] = r.times
        f["devices_strings"] = [str for (str, uuid) in r.devices]
        f["devices_uuids"] = [string(uuid) for (str, uuid) in r.devices]
        for key in keys(r.results)
            # store Vector{Vector{T}} as Matrix{T} where
            # different rows correspond to different devices
            data = reduce(hcat, r.results[key])
            f[string(key)] = data
        end
    end
    return nothing
end

"""
Given an HDF5 file created with [`save_monitoring_results`](@ref),
restore the saved monitoring results (i.e. output of [`monitoring_stop`](@ref)).
"""
function load_monitoring_results(filename::String)
    if !isfile(filename)
        error("Input file $filename doesn't exists.")
    end

    h5open(filename, "r") do f
        times = read(f["times"])
        dev_strs = read(f["devices_strings"])
        dev_uuids = read(f["devices_uuids"])
        devices = Tuple{String,UUID}[
            (dev_strs[i], UUID(dev_uuids[i])) for i in eachindex(dev_uuids)
        ]
        d = Dict{Symbol,Vector{Vector{Float64}}}()
        for key in keys(f)
            if key == "times" || key == "devices"
                continue
            end
            data = read(f[key])
            if data isa Matrix
                d[Symbol(key)] = [collect(col) for col in eachcol(data)]
            end
        end
        return MonitoringResults(times, devices, d)
    end
end
