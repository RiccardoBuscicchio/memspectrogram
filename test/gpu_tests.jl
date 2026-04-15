"""
GPU extension tests for the Memspectrum package.

Tests the MemspectrumCUDAExt extension.  Always verifies that the CPU-fallback
paths (`use_gpu=false`) work correctly when CUDA is loaded.  GPU-specific paths
(`use_gpu=true`) are only exercised when a functional CUDA device is detected at
run time; otherwise those test-sets are skipped with an informational message.

Run directly with CUDA already in the project environment:

    julia --project=. test/gpu_tests.jl

Or add it to the CI matrix.
"""

using Test
using Random
using CUDA   # triggers automatic loading of MemspectrumCUDAExt

# Load Memspectrum from source (works whether installed or not)
if !isdefined(Main, :Memspectrum)
    push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
    include(joinpath(@__DIR__, "..", "src", "Memspectrum.jl"))
end
using .Memspectrum

@testset "MemspectrumCUDAExt" begin

    Random.seed!(42)
    N  = 512
    dt = 1.0 / 256.0
    x  = randn(N)
    m  = MESA()
    solve!(m, x; verbose=false)

    # ------------------------------------------------------------------
    # CPU-fallback paths (use_gpu=false) – always run, no GPU needed
    # ------------------------------------------------------------------
    @testset "forecast CPU fallback (use_gpu=false)" begin
        preds = forecast(m, x, 50; number_of_simulations=5, use_gpu=false)
        @test size(preds) == (5, 50)
        @test all(isfinite.(preds))
    end

    @testset "mesa_spectrogram CPU fallback (use_gpu=false)" begin
        t, f, S = mesa_spectrogram(x, dt;
                                   segment_length=128,
                                   overlap=0.5,
                                   use_gpu=false)
        @test length(t) > 0
        @test length(f) == 128 ÷ 2
        @test size(S, 1) == 128 ÷ 2
        @test all(S .> 0)
    end

    # ------------------------------------------------------------------
    # Real GPU paths (use_gpu=true) – only when a CUDA GPU is present
    # ------------------------------------------------------------------
    if CUDA.functional()
        @testset "forecast GPU (use_gpu=true)" begin
            preds = forecast(m, x, 50; number_of_simulations=8, use_gpu=true)
            @test size(preds) == (8, 50)
            @test all(isfinite.(preds))
        end

        @testset "mesa_spectrogram GPU (use_gpu=true)" begin
            t, f, S = mesa_spectrogram(x, dt;
                                       segment_length=128,
                                       overlap=0.5,
                                       use_gpu=true)
            @test length(t) > 0
            @test length(f) == 128 ÷ 2
            @test size(S, 1) == 128 ÷ 2
            @test all(S .> 0)
        end
    else
        @info "No functional CUDA GPU detected; GPU-specific tests skipped."
    end

end # @testset "MemspectrumCUDAExt"
