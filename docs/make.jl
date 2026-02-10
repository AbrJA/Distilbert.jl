using Documenter, DistilBERT

makedocs(
    modules=[DistilBERT],
    sitename="DistilBERT.jl",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true",
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(
    repo="github.com/AbrJA/DistilBERT.jl.git",
)
