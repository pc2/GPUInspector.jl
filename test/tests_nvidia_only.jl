@testset "toggle_tensorcoremath" begin
    @test isnothing(CUDAExt.toggle_tensorcoremath(true; verbose=false))
    @test CUDA.math_mode() == CUDA.FAST_MATH
    @test isnothing(CUDAExt.toggle_tensorcoremath(false; verbose=false))
    @test CUDA.math_mode() == CUDA.DEFAULT_MATH
    # test toggle
    @test isnothing(CUDAExt.toggle_tensorcoremath(; verbose=false))
    @test CUDA.math_mode() == CUDA.FAST_MATH
    @test isnothing(CUDAExt.toggle_tensorcoremath(; verbose=false))
    @test CUDA.math_mode() == CUDA.DEFAULT_MATH
end
