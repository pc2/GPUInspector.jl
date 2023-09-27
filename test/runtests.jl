using GPUInspector
using Test
using LinearAlgebra
using Logging

# Environment variables:
#   - "TEST_BACKEND": can be set to manually specify a backend
#   - "TEST_QUIET": can be set to true/false to enable/disable non-verbose testing
#   - "TESTS": a comma separated list of test suites to run (see TEST_NAMES below)

# figure out which backend to use (if both CUDA and AMDGPU are functional we use CUDA)
if haskey(ENV, "TEST_BACKEND")
    if lowercase(ENV["TEST_BACKEND"]) in ("nvidia", "cuda", "nvidiabackend")
        using CUDA
        TEST_BACKEND = NVIDIABackend()
    elseif lowercase(ENV["TEST_BACKEND"]) in ("amd", "amdgpu", "amdbackend")
        using AMDGPU
        TEST_BACKEND = AMDBackend()
    else
        error("""
        TEST_BACKEND environment variable contains unsupported value.
        """)
    end
else
    using CUDA
    using AMDGPU
    if CUDA.functional()
        @info("NVIDIA GPUs detected.", CUDA.devices())
        TEST_BACKEND = NVIDIABackend()
    elseif AMDGPU.functional()
        @info("AMD GPUs detected.", AMDGPU.devices())
        TEST_BACKEND = AMDBackend()
    else
        error("""
            Aborting because neither CUDA.jl nor AMDGPU.jl are functional.
            Are there any GPUs in the system?
            """)
    end
end
backend!(TEST_BACKEND)
@info "Running tests with the following backend: $TEST_BACKEND."

const TEST_NAMES = [
    "bandwidth", "peakflops", "stresstest", "gpuinfo", "utility", "backend_specific", "core"
]
if haskey(ENV, "TESTS")
    tests = split(ENV["TESTS"], ",")
    if !all(t -> t in TEST_NAMES, tests)
        error("""
        TESTS environment variable contains unknown test names.
        Valid test names are: $(TEST_NAMES)
        """)
    end
    TARGET_TESTS = tests
else
    # run all tests
    const TARGET_TESTS = TEST_NAMES
end
@info "Running following tests: $TARGET_TESTS."

if "stresstest" in TARGET_TESTS
    # error if we aren't running with enough threads
    if Threads.nthreads() == 1 || (Threads.nthreads() < ngpus() + 1)
        # we should have at least one thread per gpu + one monitoring thread
        error("You should run the tests with at least $(ngpus() + 1) Julia threads.")
    end
end

quiet_testing = parse(Bool, get(ENV, "TEST_QUIET", "true"))
if quiet_testing
    GPUInspector.DEFAULT_IO[] = Base.BufferStream()
    global_logger(Logging.NullLogger())
end

if "core" in TARGET_TESTS
    include("tests_core.jl")
end
if "utility" in TARGET_TESTS
    include("tests_utility.jl")
end
if "gpuinfo" in TARGET_TESTS
    include("tests_gpuinfo.jl")
end
if "bandwidth" in TARGET_TESTS
    include("tests_bandwidth.jl")
end
if "stresstest" in TARGET_TESTS
    using CairoMakie
    include("tests_stresstest.jl")
end
if "peakflops" in TARGET_TESTS
    include("tests_peakflops.jl")
end
if "backend_specific" in TARGET_TESTS
    if TEST_BACKEND == NVIDIABackend()
        include("tests_nvidia_only.jl")
    elseif TEST_BACKEND == AMDBackend()
        include("tests_amd_only.jl")
    end
end
