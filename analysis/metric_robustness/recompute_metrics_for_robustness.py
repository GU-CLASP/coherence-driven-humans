import pandas as pd


BOOKKEEPING_METRICS = {"count", "observed_rows", "total_rows", "n_stories"}
PROMPT_ORDER = {"original": 0, "large": 1}
ANALYSIS_DIRNAME = "metric_exclusion_sensitivity"
ACTIVE_POLICY = "zero"


def prompt_sort_key(prompt):
    p = str(prompt)
    return (PROMPT_ORDER.get(p, 999), p)


def find_project_root(start):
    for p in [start, *start.parents]:
        if (p / "analysis" / "analysis_data").exists():
            return p
    raise FileNotFoundError("Could not locate project root containing analysis/analysis_data")


def drop_story_ids(df, exclude_ids):
    if "story_id" not in df.columns:
        return df.copy()
    sid = pd.to_numeric(df["story_id"], errors="coerce")
    return df.loc[~sid.isin(exclude_ids)].copy()


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_rank_rows(merged, rel_path, metric):
    rows = []
    for prompt in sorted(merged["prompt"].unique(), key=prompt_sort_key):
        group = merged[merged["prompt"] == prompt][["model", "prompt", f"{metric}_baseline", f"{metric}_filtered"]].copy()
        group = group.rename(columns={f"{metric}_baseline": "baseline", f"{metric}_filtered": "filtered"})

        baseline = group.sort_values("baseline", ascending=False).reset_index(drop=True)
        baseline["baseline_rank"] = range(1, len(baseline) + 1)
        filtered = group.sort_values("filtered", ascending=False).reset_index(drop=True)
        filtered["filtered_rank"] = range(1, len(filtered) + 1)

        ranked = baseline[["model", "prompt", "baseline", "baseline_rank"]].merge(
            filtered[["model", "prompt", "filtered", "filtered_rank"]], on=["model", "prompt"]
        )
        ranked["rank_shift"] = ranked["filtered_rank"] - ranked["baseline_rank"]

        for _, row in ranked.iterrows():
            rows.append(
                {
                    "file": rel_path,
                    "metric": metric,
                    "prompt": prompt,
                    "model": row["model"],
                    "baseline": row["baseline"],
                    "filtered": row["filtered"],
                    "baseline_rank": int(row["baseline_rank"]),
                    "filtered_rank": int(row["filtered_rank"]),
                    "rank_shift": int(row["rank_shift"]),
                }
            )
    return rows

def recompute_aggregates(project_root, exclude_ids, scenario_name):
    exclude_set = {int(x) for x in exclude_ids}

    base = project_root / "analysis" / "analysis_data"
    out_root = project_root / "analysis" / ANALYSIS_DIRNAME / scenario_name / "analysis_data"
    out_root.mkdir(parents=True, exist_ok=True)

    policy = ACTIVE_POLICY

    df = pd.read_csv(base / "character" / f"character_metrics_per_character_{policy}_policy.csv")
    df = drop_story_ids(df, exclude_set)
    agg = (
        df.groupby(["model", "prompt"], as_index=False)
        .agg(
            CharTr=("CharTr", "mean"),
            CharRe=("CharRe", "mean"),
            char_coherence_mean=("char_coherence", "mean"),
            char_coherence_std=("char_coherence", "std"),
            observed_rows=("char_coherence", "count"),
            total_rows=("char_coherence", "size"),
        )
    )
    agg["missing_policy"] = policy
    agg["unit"] = "character"
    agg = agg[
        [
            "model",
            "prompt",
            "CharTr",
            "CharRe",
            "char_coherence_mean",
            "char_coherence_std",
            "observed_rows",
            "total_rows",
            "missing_policy",
            "unit",
        ]
    ]
    save_csv(agg, out_root / "character" / f"character_metrics_agg_per_character_{policy}_policy.csv")

    df = pd.read_csv(base / "coreference" / "coref_metrics.csv")
    df = drop_story_ids(df, exclude_set)
    agg = (
        df.groupby(["model", "prompt"], as_index=False)
        .agg(
            num_chains=("num_chains", "mean"),
            avg_chain_size=("avg_chain_size", "mean"),
            coref_ratio_mean=("coref_ratio", "mean"),
            coref_ratio_std=("coref_ratio", "std"),
            count=("coref_ratio", "size"),
        )
    )
    save_csv(agg, out_root / "coreference" / "coref_metrics_agg.csv")

    df = pd.read_csv(base / "groovist" / "groovist_metrics_per_story.csv")
    df = drop_story_ids(df, exclude_set)
    agg = (
        df.groupby(["model", "prompt"], as_index=False)
        .agg(
            groovist_raw_mean=("groovist_raw", "mean"),
            groovist_mean=("groovist", "mean"),
            groovist_std=("groovist", "std"),
            observed_rows=("groovist", "count"),
            total_rows=("groovist", "size"),
        )
    )
    agg["unit"] = "story"
    agg["missing_policy"] = "all"
    agg = agg[
        [
            "model",
            "prompt",
            "groovist_raw_mean",
            "groovist_mean",
            "groovist_std",
            "observed_rows",
            "total_rows",
            "unit",
            "missing_policy",
        ]
    ]
    save_csv(agg, out_root / "groovist" / "groovist_metrics_agg.csv")

    df = pd.read_csv(base / "implicit_connectives" / "discourse_metrics.csv")
    df = drop_story_ids(df, exclude_set)
    agg = (
        df.groupby(["model", "prompt"], as_index=False)
        .agg(
            n_unique_relations=("n_unique_relations", "mean"),
            n_total_relations=("n_total_relations", "mean"),
            n_none_relations=("n_none_relations", "mean"),
            none_rate=("none_rate", "mean"),
            discourse_diversity_mean=("discourse_diversity", "mean"),
            discourse_diversity_std=("discourse_diversity", "std"),
            count=("discourse_diversity", "size"),
        )
    )
    save_csv(agg, out_root / "implicit_connectives" / "discourse_metrics_agg.csv")

    df = pd.read_csv(base / "mcc" / f"mcc_metrics_per_character_{policy}_policy.csv")
    df = drop_story_ids(df, exclude_set)
    agg = (
        df.groupby(["model", "prompt"], as_index=False)
        .agg(
            tc_mean=("tc", "mean"),
            tc_std=("tc", "std"),
            ic_mean=("ic", "mean"),
            ic_std=("ic", "std"),
            mcc_raw_mean=("mcc_raw_char", "mean"),
            MCC_mean=("MCC_char", "mean"),
            MCC_std=("MCC_char", "std"),
            observed_rows=("MCC_char", "count"),
            total_rows=("MCC_char", "size"),
        )
    )
    agg["missing_policy"] = policy
    agg["unit"] = "character"
    agg = agg[
        [
            "model",
            "prompt",
            "tc_mean",
            "tc_std",
            "ic_mean",
            "ic_std",
            "mcc_raw_mean",
            "MCC_mean",
            "MCC_std",
            "observed_rows",
            "total_rows",
            "missing_policy",
            "unit",
        ]
    ]
    save_csv(agg, out_root / "mcc" / f"mcc_metrics_agg_per_character_{policy}_policy.csv")

    df = pd.read_csv(base / "mci" / f"mci_ratio_metrics_per_story_{policy}_policy.csv")
    df = drop_story_ids(df, exclude_set)
    agg = (
        df.groupby(["model", "prompt"], as_index=False)
        .agg(
            groovist_mean=("groovist", "mean"),
            MCC_mean=("MCC", "mean"),
            mci_tanh_mcc_over_gv_mean=("mci_tanh_mcc_over_gv", "mean"),
            mci_tanh_gv_over_mcc_mean=("mci_tanh_gv_over_mcc", "mean"),
            observed_rows=("mci_tanh_mcc_over_gv", "count"),
            total_rows=("mci_tanh_mcc_over_gv", "size"),
        )
    )
    agg["missing_policy"] = policy
    agg["unit"] = "story"
    agg = agg[
        [
            "model",
            "prompt",
            "groovist_mean",
            "MCC_mean",
            "mci_tanh_mcc_over_gv_mean",
            "mci_tanh_gv_over_mcc_mean",
            "observed_rows",
            "total_rows",
            "missing_policy",
            "unit",
        ]
    ]
    save_csv(agg, out_root / "mci" / f"mci_ratio_metrics_agg_{policy}_policy.csv")

    df = pd.read_csv(base / "ncs" / "ncs_per_story_zero_policy.csv")
    df = drop_story_ids(df, exclude_set)
    save_csv(df, out_root / "ncs" / "ncs_per_story_zero_policy.csv")
    agg = (
        df.groupby(["model", "prompt"], as_index=False)
        .agg(
            n_stories=("story_id", "size"),
            ncs_arithmetic_mean=("ncs_arithmetic", "mean"),
            ncs_arithmetic_std=("ncs_arithmetic", "std"),
            ncs_geometric_mean=("ncs_geometric", "mean"),
            ncs_geometric_std=("ncs_geometric", "std"),
        )
    )
    save_csv(agg, out_root / "ncs" / "ncs_aggregate_zero_policy.csv")

    df = pd.read_csv(base / "topic_modelling" / "topic_switch_metrics.csv")
    df = drop_story_ids(df, exclude_set)
    agg = (
        df.groupby(["model", "prompt"], as_index=False)
        .agg(
            n_sentences=("n_sentences", "mean"),
            topic_switch_rate_mean=("topic_switch_rate", "mean"),
            topic_switch_rate_std=("topic_switch_rate", "std"),
            count=("topic_switch_rate", "size"),
        )
    )
    save_csv(agg, out_root / "topic_modelling" / "topic_switch_metrics_agg.csv")

    return out_root


def compare_against_baseline(project_root, scenario_root):
    base = project_root / "analysis" / "analysis_data"

    targets = [
        ("character/character_metrics_agg_per_character_zero_policy.csv", ["char_coherence_mean", "observed_rows", "total_rows"]),
        ("coreference/coref_metrics_agg.csv", ["coref_ratio_mean", "count"]),
        ("groovist/groovist_metrics_agg.csv", ["groovist_mean", "observed_rows", "total_rows"]),
        ("implicit_connectives/discourse_metrics_agg.csv", ["discourse_diversity_mean", "count"]),
        ("mcc/mcc_metrics_agg_per_character_zero_policy.csv", ["MCC_mean", "observed_rows", "total_rows"]),
        ("mci/mci_ratio_metrics_agg_zero_policy.csv", ["mci_tanh_mcc_over_gv_mean", "observed_rows", "total_rows"]),
        ("topic_modelling/topic_switch_metrics_agg.csv", ["topic_switch_rate_mean", "count"]),
        
        ("ncs/ncs_aggregate_zero_policy.csv", ["ncs_arithmetic_mean", "ncs_geometric_mean", "n_stories"]),
    ]

    delta_rows = []
    rank_rows = []

    for rel_path, metrics in targets:
        base_df = pd.read_csv(base / rel_path)
        scenario_df = pd.read_csv(scenario_root / rel_path)
        keys = ["model", "prompt"]
        common_metrics = [m for m in metrics if m in base_df.columns and m in scenario_df.columns]

        merged = base_df[keys + common_metrics].merge(
            scenario_df[keys + common_metrics], on=keys, suffixes=("_baseline", "_filtered")
        )

        for _, row in merged.iterrows():
            for metric in common_metrics:
                b = row[f"{metric}_baseline"]
                r = row[f"{metric}_filtered"]
                delta_rows.append(
                    {
                        "file": rel_path,
                        "model": row["model"],
                        "prompt": row["prompt"],
                        "metric": metric,
                        "baseline": b,
                        "filtered": r,
                        "delta": r - b,
                    }
                )

        for metric in common_metrics:
            if metric in BOOKKEEPING_METRICS:
                continue
            rank_rows.extend(build_rank_rows(merged, rel_path, metric))

    deltas = pd.DataFrame(delta_rows)
    ranks = pd.DataFrame(rank_rows)
    return deltas, ranks


def run_scenario(project_root, label, exclude_ids):
    scenario_name = f"{label}_excl_{'_'.join(str(x) for x in sorted(exclude_ids))}"
    scenario_root = recompute_aggregates(project_root, exclude_ids, scenario_name)
    deltas, ranks = compare_against_baseline(project_root, scenario_root)

    comp_dir = scenario_root.parent / "comparison"
    deltas_path = comp_dir / "baseline_vs_exclusion_deltas.csv"
    ranks_path = comp_dir / "rank_changes_by_prompt.csv"

    save_csv(deltas, deltas_path)
    save_csv(ranks, ranks_path)

    return {
        "label": label,
        "exclude_ids": sorted(exclude_ids),
        "out_root": scenario_root,
        "deltas_path": deltas_path,
        "ranks_path": ranks_path,
    }
