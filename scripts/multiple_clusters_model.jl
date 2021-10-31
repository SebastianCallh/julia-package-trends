include("../src/common.jl")
using ReverseDiff, Memoization, BSON
Turing.setadbackend(:reversediff)
Turing.emptyrdcache()
Turing.setrdcache(true)

out_path = joinpath(plots_path, "multiple-clusters-model")
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

function posterior_predictive_plot(plt, chain, t)
    pred = prediction(chain, t)
    for r in unique(rr)[1:3]
        r_mask = rr .== r
        p = pred[:, r_mask, 1]
        ss = summarystats(group(p, "y"))
        plot!(
            plt,
            day_transform(xx; inverse=true)[r_mask],
            ss[:,:mean],
            ribbon=2*ss[:,:std],
            alpha=0.3,
            title="Posterior predictive distribution",
            xlabel="Days",
            ylabel="Package downloads (log10)",
            label=nothing,
            legend=:bottomright,
            color=r,
            linewidth=4
        )
    end  
    plt
end

@model multiple_clusters_model(x, y, r, R, t, T) = begin
    αᵣμ ~ Normal(2, 1)
    βᵣμ ~ Normal(0, 1)
    αₜμ ~ Normal(2, 1)
    βₜμ ~ Normal(0, 1)

    αᵣoff ~ filldist(Normal(), R)
    αᵣscale ~ TruncatedNormal(0, 1, 0, Inf)
    βᵣoff ~ filldist(Normal(), R)
    βᵣscale ~ TruncatedNormal(0, 1, 0, Inf)

    αₜoff ~ filldist(Normal(), T)
    αₜscale ~ TruncatedNormal(0, 1, 0, Inf)
    βₜoff ~ filldist(Normal(), T)
    βₜscale ~ TruncatedNormal(0, 1, 0, Inf)

    α ~ Normal(1, 1.5)
    β ~ Normal(0, 1)
    σ ~ filldist(filldist(TruncatedNormal(0, 1, 0, Inf), R), T)

    αₜ ~ lazydist(Dirac, αₜμ .+ αₜoff .* αₜscale)
    βₜ ~ lazydist(Dirac, βₜμ .+ βₜoff .* βₜscale)
    αᵣ ~ lazydist(Dirac, αᵣμ .+ αᵣoff .* αᵣscale)
    βᵣ ~ lazydist(Dirac, βᵣμ .+ βᵣoff .* βᵣscale)
    y ~ lazydist(Normal, α .+ αᵣ[r] .+ αₜ[t] .+ (β .+ βᵣ[r] .+ βₜ[t]).*x, σ[r][t])
end

const prediction(chain, t) = predict(multiple_clusters_model(xx, missing, rr, R, t, T), chain)
model = multiple_clusters_model(x, y, r, R, t, T)
prior_chain = sample(model, Prior(), 50)
prior_pooled_pred_plt = prior_predictive_plot(prior_chain, Int.(ones(size(tt))))
scatter_data(prior_pooled_pred_plt, df)
plot(prior_chain)

# takes about a minute
@time post_chain = sample(model, NUTS(), MCMCThreads(), 500, 4, progress=false);
plot(post_chain)

eu_us_df = filter(x -> x.region in ["us-east", "us-west", "eu-central"], df)
post_pred_user_plt = plot();
posterior_predictive_plot(post_pred_user_plt, post_chain, Int.(1*ones(size(tt))));
scatter_data(post_pred_user_plt, filter(x -> x.client_type == "user", eu_us_df));
plot!(post_pred_user_plt, title="Posterior predictive, users")

post_pred_ci_plt = plot();
posterior_predictive_plot(post_pred_ci_plt, post_chain, Int.(2*ones(size(tt))));
scatter_data(post_pred_ci_plt, filter(x -> x.client_type == "ci", eu_us_df))
plot!(post_pred_ci_plt, title="Posterior predictive, CI")
post_pred_plt = plot(post_pred_user_plt, post_pred_ci_plt, layout=(2, 1), size=(600, 600))
savefig(post_pred_plt, save_path("posterior_predictive_us_eu.svg"))

# this model assumes same deviation from the "global + regional" mean per 
# client type, which doesn't hold. Below are differences per client type
# per region, which isn't exactly the same (since we have no model) but is at least a proxy
let
    region_subset = Set(["us-east", "us-west", "eu-central"])
    df′ = filter(x -> x.region in region_subset, df)

    requests_plt = plot(
        title="Requests per region and client type",
        xlabel = "Package requests (log10)",
        ylabel = "Density"
    );
    linestyles = [:solid, :dash]
    for (key, group) in pairs(groupby(df′, [:client_type_index, :region_index]))
        density!(
            requests_plt, group.log10_request_count,
            color=colors[key.region_index],
            linestyle=linestyles[key.client_type_index],
            label="$(regions[key.region_index]) - $(client_types[key.client_type_index])",
            legend=:topleft
        )   
    end

    diff_plot = plot(title="Requests difference between client types")
    for (key, group) in pairs(groupby(df′, :region_index))
        df_user = filter(x -> x.client_type == "user", group)
        df_ci = filter(x -> x.client_type == "ci", group)
        n = 5000
        diff = rand(df_user.log10_request_count, n) .- rand(df_ci.log10_request_count, n)

        density!(
            diff_plot, diff,
            color=colors[key.region_index],
            label="$(regions[key.region_index])",
            legend=nothing
        )   
    end

    plt = plot(requests_plt, diff_plot, layout=(2, 1), size=(500, 500))
    savefig(plt, save_path("request_client_type_diff_per_region.svg"))
end