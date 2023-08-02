module CairoMakieExt

using GPUInspector
import GPUInspector: MonitoringResults, _defaultylims, _symbol2title_and_label
using CairoMakie

function GPUInspector.savefig_monitoring_results(r::MonitoringResults, symbols=keys(r.results); ext=:pdf)
    for s in symbols
        savefig_monitoring_results(r, s; ext)
    end
    return nothing
end

function GPUInspector.savefig_monitoring_results(r::MonitoringResults, s::Symbol; ext=:pdf)
    times = r.times
    values = r.results[s]
    title, ylabel = _symbol2title_and_label(s)
    ylims = _defaultylims(values)
    device_labels = [str for (str, uuid) in r.devices]

    f = CairoMakie.Figure(; resolution=(1000, 500))
    ax =
        f[1, 1] = CairoMakie.Axis(
            f; xlabel="Time [s]", ylabel=ylabel, title=title, ylims=ylims
        )
    CairoMakie.scatterlines!(times, getindex.(values, 1); label=device_labels[1])
    for i in 2:length(first(values))
        CairoMakie.scatterlines!(times, getindex.(values, i); label=device_labels[i])
    end
    f[1, 2] = CairoMakie.Legend(f, ax, "Devices"; framevisible=false)
    filename =
        replace(replace(replace(lowercase(title), " " => "_"), "(" => ""), ")" => "") *
        "_plot.$(string(ext))"
    CairoMakie.save(filename, f)
    return nothing
end

end # module
