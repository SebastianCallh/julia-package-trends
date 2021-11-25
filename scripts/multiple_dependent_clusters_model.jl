include("../src/common.jl")
using ReverseDiff, Memoization, BSON
Turing.setadbackend(:reversediff)
Turing.emptyrdcache()
Turing.setrdcache(true)

out_path = joinpath(plots_path, "multiple-dependent--clusters-model")
isdir(out_path) || mkdir(out_path)
save_path(file) = joinpath(out_path, file)

const M = 50
const N = 50
const xx = mapreduce(r -> range(extrema(x)..., length=N), vcat, 1:R)
const rr = mapreduce(r -> repeat([r], N), vcat, 1:R)
const tt = mapreduce(t -> repeat([t], N), vcat, repeat(collect(1:T), 3))

function prior_predictive_plot(chain, t)
    pred = prediction(chain, t)
    plot(
        day_transform(xx; inverse=true),
        group(pred, "y").value.data[:,:,1]',
        group=rr,
        title="Prior samples",
        xlabel="Days",
        ylabel="Package downloads (log10)",
        label=nothing,
        color=:blue,
        alpha=0.2
    )
end

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

@model multiple_clusters_model(x, y, r, R, t, T) = begin
    region_type_index = Dict((r, t) => i for (i, (r, t)) in enumerate(Base.Iterators.product(1:R, 1:T)))
    rt = [region_type_index[rt] for rt in zip(r, t)]
    RT = length(region_type_index)
    
    αᵣμ ~ Normal(0, 1)
    βᵣμ ~ Normal(0, 1)
    αᵣₜμ ~ Normal(0, 1)
    βᵣₜμ ~ Normal(0, 1)

    αᵣoff ~ filldist(Normal(), R)
    αᵣscale ~ TruncatedNormal(0, 0.2, 0, Inf)
    βᵣoff ~ filldist(Normal(), R)
    βᵣscale ~ TruncatedNormal(0, 0.2, 0, Inf)

    αᵣₜoff_mu ~ Normal()
    αᵣₜoff_scale ~ TruncatedNormal(0, 0.2, 0, Inf)
    αᵣₜoff ~ filldist(Normal(αᵣₜoff_mu, αᵣₜoff_scale), RT)
    αᵣₜscale ~ TruncatedNormal(0, 1, 0, Inf)

    βᵣₜoff_mu ~ Normal()
    βᵣₜoff_scale ~ TruncatedNormal(0, 0.2, 0, Inf)
    βᵣₜoff ~ filldist(Normal(βᵣₜoff_mu, βᵣₜoff_scale), RT)
    βᵣₜscale ~ TruncatedNormal(0, 0.2, 0, Inf)
   
    α ~ Normal(4, 1.5)
    β ~ Normal(0, 0.5)
    σ ~ filldist(TruncatedNormal(0, 1, 0, Inf), RT)
    
    αᵣ ~ lazydist(Dirac, αᵣμ .+ αᵣoff .* αᵣscale)
    βᵣ ~ lazydist(Dirac, βᵣμ .+ βᵣoff .* βᵣscale)
    αᵣₜ ~ lazydist(Dirac, αᵣₜμ .+ αᵣₜoff .* αᵣₜscale)
    βᵣₜ ~ lazydist(Dirac, βᵣₜμ .+ βᵣₜoff .* βᵣₜscale)
    y ~ lazydist(Normal, α .+ αᵣ[r] .+ αᵣₜ[rt] .+ (β .+ βᵣ[r] .+ βᵣₜ[rt]).*x, σ[rt])
end

const prediction(chain, t) = predict(multiple_clusters_model(xx, missing, rr, R, t, T), chain)
model = multiple_clusters_model(x, y, r, R, t, T)
prior_chain = sample(model, Prior(), 50)
prior_pooled_pred_plt = prior_predictive_plot(prior_chain, Int.(ones(size(tt))))
scatter_data(prior_pooled_pred_plt, df)
plot(prior_chain)

# takes about three minutes
@time post_chain = sample(model, NUTS(), MCMCThreads(), 500, 8, progress=false);
plot(post_chain)

eu_us_df = filter(x -> x.region in ["us-east", "us-west", "eu-central"], df)
post_pred_user_plt = plot();
posterior_predictive_plot(post_pred_user_plt, post_chain, unique(rr), rr, Int.(1*ones(size(tt))));
scatter_data(post_pred_user_plt, filter(x -> x.client_type == "user", df));
plot!(post_pred_user_plt, title="Posterior predictive, users", ylim=(0.5, 6))

post_pred_ci_plt = plot();
posterior_predictive_plot(post_pred_ci_plt, post_chain, unique(rr), rr, Int.(2*ones(size(tt))));
scatter_data(post_pred_ci_plt, filter(x -> x.client_type == "ci", df))
plot!(post_pred_ci_plt, title="Posterior predictive, CI")
post_pred_plt = plot(post_pred_user_plt, post_pred_ci_plt, layout=(2, 1), size=(600, 800))
savefig(post_pred_plt, save_path("posterior_predictive_us_eu.svg"))