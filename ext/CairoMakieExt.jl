module CairoMakieExt

using GPUInspector
import GPUInspector: MonitoringResults, _defaultylims, _symbol2title_and_label, savefig_monitoring_results
using CairoMakie

function savefig_monitoring_results(
    r::MonitoringResults, symbols=keys(r.results); ext=:pdf, prefix=""
)
    if symbols isa Symbol
        _savefig_monitoring_results(r, symbols; ext, prefix)
    else
        for s in symbols
            _savefig_monitoring_results(r, s; ext, prefix)
        end
    end
    return nothing
end

function savefig_monitoring_results(filename::String, r::MonitoringResults; symbol=nothing)
    if isnothing(symbol)
        # If no symbol is provided, we create a tiled summary of ALL results
        _savefig_monitoring_results_summary(r; filename)
    else
        _savefig_monitoring_results(r, symbol; filename)
    end
    return nothing
end

function _savefig_monitoring_results(r::MonitoringResults, s::Symbol; ext=:pdf, filename=nothing, prefix="")
    times = r.times
    values = r.results[s]
    title, ylabel = _symbol2title_and_label(s)
    ylims = _defaultylims(values)
    device_labels = [str for (str, uuid) in r.devices]

    f = CairoMakie.Figure(; size=(1000, 500))
    _plot_metric!(f[1, 1], times, values, title, ylabel, ylims, device_labels)
    
    if isnothing(filename)
        clean_title = replace(replace(replace(lowercase(title), " " => "_"), "(" => ""), ")" => "")
        filename = prefix * clean_title * "_plot.$(string(ext))"
    end
    CairoMakie.save(filename, f)
    return nothing
end

function _savefig_monitoring_results_summary(r::MonitoringResults; filename="monitoring_summary.pdf")
    symbols = collect(keys(r.results))
    n = length(symbols)
    times = r.times
    device_labels = [str for (str, uuid) in r.devices]

    # Create a tall figure to fit all plots
    f = CairoMakie.Figure(; size=(1000, 400 * n))
    
    for (i, s) in enumerate(symbols)
        values = r.results[s]
        title, ylabel = _symbol2title_and_label(s)
        ylims = _defaultylims(values)
        
        # Plot into the i-th row
        _plot_metric!(f[i, 1], times, values, title, ylabel, ylims, device_labels; show_legend=(i==1))
    end
    
    CairoMakie.save(filename, f)
    return nothing
end

function _plot_metric!(target, times, values, title, ylabel, ylims, device_labels; show_legend=true)
    ax = CairoMakie.Axis(target; xlabel="Time [s]", ylabel=ylabel, title=title)
    CairoMakie.ylims!(ax, ylims)
    
    # Use a color cycle for devices
    for i in 1:length(first(values))
        CairoMakie.scatterlines!(ax, times, getindex.(values, i); label=device_labels[i])
    end
    
    if show_legend && length(device_labels) > 0
        # Position legend based on whether it's a single plot or tiled
        # For simplicity in tiled, we'll put it to the right of the first plot
        # but CairoMakie figures handle this via Layout positions
    end
    return ax
end

end # module
