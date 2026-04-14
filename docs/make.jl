using Documenter
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "Memspectrum.jl"))
using .Memspectrum

makedocs(
    modules  = [Memspectrum],
    sitename = "Memspectrum.jl",
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    pages = [
        "Home"     => "index.md",
        "API"      => "api.md",
        "Examples" => "examples.md",
    ],
)

deploydocs(
    repo   = "github.com/RiccardoBuscicchio/memspectrogram.git",
    branch = "gh-pages",
    push_preview = true,
)
