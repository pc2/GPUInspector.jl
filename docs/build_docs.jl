# println("--- :julia: Instantiating project")
# using Pkg
# Pkg.activate("..")
# Pkg.instantiate()
# Pkg.activate(".")
# Pkg.instantiate()
# push!(LOAD_PATH, joinpath(@__DIR__, ".."))
# println("+++ :julia: Building documentation")
# include("make.jl")

using Pkg
@info "Instantiating doc environment"
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()
@info "Building documentation"
include(joinpath(@__DIR__, "make.jl"))
