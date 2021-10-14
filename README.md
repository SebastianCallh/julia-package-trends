# Julia packages download stats
This repo hosts a Bayesian analysis of package trends in the julia ecosystem. 
It currently uses a two-level model where regional trends are drawn from a global trend. 
This model is then used independently for CI requests and end user requests.

Main takeaways include
- Package download trend is overall stable
- User downloads are increasing in the US
- User downloads are constant in Europe
- User downloads are decreasing in China
- CI downloads are increasing slightly

## Acquiring the data
Links to downloading the data can be found [here](https://discourse.julialang.org/t/announcing-package-download-stats/69073). This repo makes use of the `package_requests_by_region_by_date.csv` file, so download it and extract into the data directory and you are good to go.

## Reproducing the analysis
After acquiring the data, drop into a Julia shell with `julia --project`, instantiate the project by hitting `]` followed by `instantiate RET`, and then run `include("scripts\produce_plots.jl")` to reproduce the plots.

![posterior predictive distribution for CI requests](./plots/ci/package_request_posterior_predictive.svg)