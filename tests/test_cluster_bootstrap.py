import pandas as pd

from eval_tool.aggregate import bootstrap_cluster_mean_ci, bootstrap_mean_ci, summarize_binary_metric


def test_one_row_per_cluster_is_exactly_the_row_bootstrap():
    values = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    clusters = [f"img{i}" for i in range(len(values))]
    assert bootstrap_cluster_mean_ci(values, clusters) == bootstrap_mean_ci(values)


def test_correlated_rows_widen_the_interval():
    # 20 images x 5 questions; an image is either all right or all wrong
    values = [int(i % 2 == 0) for i in range(20) for _ in range(5)]
    clusters = [f"img{i}" for i in range(20) for _ in range(5)]
    row_low, row_high = bootstrap_mean_ci(values)
    low, high = bootstrap_cluster_mean_ci(values, clusters)
    assert (high - low) > 1.8 * (row_high - row_low)


def test_rows_without_a_cluster_id_are_their_own_clusters():
    # derived rows carry no source_id; under pandas 3 they used to vanish from the groupby
    values = [1, 1, 0, 1, 1, 1, 0]
    low, high = bootstrap_cluster_mean_ci(values, [float("nan")] * len(values))
    assert (low, high) == bootstrap_mean_ci(values)


def test_summary_clusters_on_source_id():
    data = pd.DataFrame({
        "model": "m", "dataset": "d",
        "hit": [int(i % 2 == 0) for i in range(20) for _ in range(5)],
        "source_id": [f"img{i}" for i in range(20) for _ in range(5)],
    })
    clustered = summarize_binary_metric(data, ["model", "dataset"])
    flat = summarize_binary_metric(data.drop(columns="source_id"), ["model", "dataset"])
    width = lambda s: float(s["ci_high"].iloc[0] - s["ci_low"].iloc[0])  # noqa: E731
    assert clustered["score"].iloc[0] == flat["score"].iloc[0] == 0.5
    assert width(clustered) > 1.8 * width(flat)
