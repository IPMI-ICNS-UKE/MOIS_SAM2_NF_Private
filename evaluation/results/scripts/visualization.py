import plotly


def add_metric(aggregated_metrics, model_result, testset, metric_name, metric_key):
    values = aggregated_metrics[metric_name]
    new_values = model_result[testset]["aggregated"][metric_key]
    if values["mean"] is not None and new_values["mean"] is not None:
        values["mean"] += new_values["mean"]
        values["std"] += new_values["std"]
    else:
        values["mean"] = None
        values["std"] = None
