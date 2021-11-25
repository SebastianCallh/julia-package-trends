include("../src/common.jl")
using ReverseDiff, Memoization, BSON
Turing.setadbackend(:reversediff)
Turing.emptyrdcache()
Turing.setrdcache(true)

out_path = joinpath(plots_path, "pooled-regiontypes-model")
isdir(out_path) || mkdir(out_path)
save_path(file) = joinpath(out_path, file)

seasonal_features(x; num_features) = Matrix(DataFrame(Dict(
    string(func) * string(order) => func.(2π.*x.*order)
    for func in [sin, cos], order in 1:(num_features/2)
)))'
 

debug = DataFrame(Dict(
    string(func) * string(Int(order)) => func.(2π.*x.*order)
    for func in [sin, cos], order in 1:(10/2)
))

x̃ = df.day ./ maximum(df.day)
period = 30 # days
S = 10
s = seasonal_features(df.day ./ period; num_features=S)
ss_grid = [df.day[i] for i in Int64.(round.(range(1, size(df.day, 1), length=R*N)))]
ss = seasonal_features(ss_grid ./ period; num_features=S)

function posterior_predictive_plot(plt, chain, r_to_plot, rr, tt)
    pred = prediction(chain, tt)
    for r in filter(r -> r in r_to_plot, unique(rr))
        r_mask = rr .== r
        p = pred[:, r_mask, 1]
        ss = summarystats(group(p, "y"))
        plot!(
            plt,
            day_transform(xx; inverse=true)[r_mask],
            ss[:,:mean],
            ribbon=2*ss[:,:std],
            ribbonalpha=0.3,
            title="Posterior predictive distribution",
            xlabel="Days",
            ylabel="Package downloads (log10)",
            label=nothing,
            legend=:bottomright,
            color=r,
            linewidth=5
        )
    end
    plt
end

@model pooled_regiontypes_seasonal_model(x, y, s, S, r, R, t, T) = begin
    region_type_index = Dict((r, t) => i for (i, (r, t)) in enumerate(Base.Iterators.product(1:R, 1:T)))
    rt = [region_type_index[rt] for rt in zip(r, t)]
    RT = length(region_type_index)

    αᵣₜμμ ~ Normal(0, 1)
    αᵣₜμσ ~ TruncatedNormal(0, 1, 0, Inf)    
    αᵣₜμ ~ Normal(αᵣₜμμ, αᵣₜμσ)

    βᵣₜμμ ~ Normal(0, 1)
    βᵣₜμσ ~ TruncatedNormal(0, 1, 0, Inf)
    βᵣₜμ ~ Normal(βᵣₜμμ, βᵣₜμσ)

    αᵣₜoff_mu_mu ~ Normal()
    αᵣₜoff_mu_scale ~ TruncatedNormal(0, 0.1, 0, Inf)
    αᵣₜoff_mu ~ Normal(αᵣₜoff_mu_mu, αᵣₜoff_mu_scale)
    αᵣₜoff_scale ~ TruncatedNormal(0, 0.1, 0, Inf)
    αᵣₜoff ~ filldist(Normal(αᵣₜoff_mu, αᵣₜoff_scale), RT)
    αᵣₜscale ~ TruncatedNormal(0, 0.1, 0, Inf)

    βᵣₜoff_mu_mu ~ Normal()
    βᵣₜoff_mu_scale ~ TruncatedNormal(0, 1, 0, Inf)
    βᵣₜoff_mu ~ Normal(βᵣₜoff_mu_mu, βᵣₜoff_mu_scale)
    βᵣₜoff_scale ~ TruncatedNormal(0, 1, 0, Inf)
    βᵣₜoff ~ filldist(Normal(βᵣₜoff_mu, βᵣₜoff_scale), RT)
    βᵣₜscale ~ TruncatedNormal(0, 1, 0, Inf)

    αμ ~ Normal(4, 0.5)
    ασ ~ TruncatedNormal(0, 0.5, 0, Inf)
    α ~ Normal(αμ, ασ)

    βμ ~ Normal(0, 0.5)
    βσ ~ TruncatedNormal(0, 0.5, 0, Inf)
    β ~ Normal(βμ, βσ)
    σ ~ filldist(TruncatedNormal(0, 0.2, 0, Inf), RT)

    αᵣₜ ~ lazydist(Dirac, αᵣₜμ .+ αᵣₜoff .* αᵣₜscale)
    βᵣₜ ~ lazydist(Dirac, βᵣₜμ .+ βᵣₜoff .* βᵣₜscale)
    βₛ ~ filldist(Normal(0, 0.1), RT, S)
    #= @show size.((βₛ[rt,:], s))
    @show size(sum(βₛ[rt,:].*s'; dims=2)[:])
    @show size(βₛ[rt,:]*s) =#
    # y ~ lazydist(Normal, α .+ αᵣₜ[rt] .+ (β .+ βᵣₜ[rt]).*x .+ vec(sum(βₛ[rt,:].*s'; dims=2)), σ[rt])
    # y ~ lazydist(Normal, (α .+ αᵣₜ[rt] .+ (β .+ βᵣₜ[rt]).*x) .* (1 .+ vec(sum(βₛ[rt,:].*s'; dims=2))), σ[rt])
    # y ~ lazydist(Normal, α .+ αᵣₜ[rt] .+ (β .+ βᵣₜ[rt]).*x, σ[rt])
    y ~ lazydist(Normal, (α .+ αᵣₜ[rt]) .* (1 .+ vec(sum(βₛ[rt,:].*s'; dims=2))), σ[rt])
end

const prediction(chain, t) = predict(pooled_regiontypes_seasonal_model(xx, missing, ss, S, rr, R, t, T), chain)
model = pooled_regiontypes_seasonal_model(x, y, s, S, r, R, t, T)
prior_chain = sample(model, Prior(), 2)
prior_pooled_pred_plt = prior_predictive_plot(prior_chain, Int.(ones(size(tt))))
scatter_data(prior_pooled_pred_plt, df)
plot(prior_chain)

# @time post_chain = sample(model, NUTS(), MCMCThreads(), 500, 8, progress=false);
@time post_chain = sample(model, NUTS(), 500)
plot(post_chain)

regions_to_plot_df = df # filter(x -> x.region in ["us-east", "us-west", "eu-central"], df)
post_pred_user_plt = plot();
posterior_predictive_plot(post_pred_user_plt, post_chain, unique(rr), rr, Int.(1*ones(size(tt))));
scatter_data(post_pred_user_plt, filter(x -> x.client_type == "user", regions_to_plot_df));
plot!(post_pred_user_plt, title="Posterior predictive, users")

post_pred_ci_plt = plot();
posterior_predictive_plot(post_pred_ci_plt, post_chain, unique(rr)[1:3], rr, Int.(2*ones(size(tt))));
scatter_data(post_pred_ci_plt, filter(x -> x.client_type == "ci", regions_to_plot_df))
plot!(post_pred_ci_plt, title="Posterior predictive, CI")
post_pred_plt = plot(post_pred_user_plt, post_pred_ci_plt, layout=(2, 1), size=(600, 600))
savefig(post_pred_plt, save_path("posterior_predictive_us_eu.svg"))
