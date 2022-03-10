# function worker_create(n::Integer=1)
#         @info("Creating worker process.")
#         pids = addprocs(1, exeflags = ["-t $(length(CUDA.devices()))", "--project=$(Base.active_project())"])
#         # _load_me_on_workers(GPUInspector; pids=pids)
#         return pids
# end

# function _load_me_on_workers(m::Module=@__MODULE__; pids = workers())
#     @show m
#     # pkgid = Base.PkgId(m)
#     # @everywhere pids Base.require($pkgid)
#     @everywhere pids Base.require(Main, :GPUInspector)
# end

"""
    @worker_create n -> pids
Create `n` workers (i.e. separate Julia processes) and
execute `using GPUInspector, CUDA` on all of them.
Returns the `pids` of the created workers.
"""
macro worker_create(n)
    quote
        nt = length(CUDA.devices())
        @info("Creating $(Int($n)) worker process(es) with $nt Julia threads each.")
        pids = addprocs(
            Int($n); exeflags=["-t $(nt)", "--project=$(Base.active_project())"]
        )
        @everywhere pids using GPUInspector, CUDA
        return pids
    end
end

"Kills all Julia workers."
macro worker_killall()
    quote
        @info("Killing all worker process(es).")
        rmprocs(workers())
        @info("Done.")
        nothing
    end
end

"""
    @worker ex
Creates a worker process, spawns the given command on it,
and kills the worker process once the command has finished execution.

**Implementation:** a Julia thread (we use `@spawn`) will be used to `wait` on the task and kill the worker.

**Examples:**
```julia
@worker GPUInspector.functional()
@worker stresstest(CUDA.devices(); duration=10, verbose=false)
```
"""
macro worker(ex)
    quote
        @info("Creating worker process.")
        pid, = addprocs(
            1;
            exeflags=["-t $(length(CUDA.devices()))", "--project=$(Base.active_project())"],
        )
        @everywhere pid using GPUInspector, CUDA

        $:(; t=Distributed.@spawnat pid $(esc(ex)))

        Threads.@spawn begin
            wait(t)
            # @info("Killing worker process.")
            rmprocs(pid)
        end
        t
    end
end

"""
    @worker pid ex
Spawns the given command on the given worker process.

**Examples:**
```julia
@worker 3 GPUInspector.functional()
@worker 3 stresstest(CUDA.devices(); duration=10, verbose=false)
```
"""
macro worker(pid, ex)
    quote
        @info("Spawning on worker process $(Int($pid)).")
        $:(; t=Distributed.@spawnat $pid $(esc(ex)))
    end
end
