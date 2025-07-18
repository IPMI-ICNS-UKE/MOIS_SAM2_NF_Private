from itertools import combinations
from scipy.stats import wilcoxon


def compare_models_wilcoxon(results_by_model, alpha=0.05, correction='bonferroni'):
    """
    Perform pairwise Wilcoxon signed-rank tests across models.

    Args:
        results_by_model (dict): {model_name: list of scan-level metrics}.
        alpha (float): Significance threshold.
        correction (str): 'bonferroni' or None.

    Returns:
        List[dict]: List of comparison results.
    """
    model_names = list(results_by_model.keys())
    num_comparisons = len(list(combinations(model_names, 2)))
    results = []

    for model_a, model_b in combinations(model_names, 2):
        scores_a = results_by_model[model_a]
        scores_b = results_by_model[model_b]

        try:
            stat, p_value = wilcoxon(scores_a, scores_b)
        except ValueError:
            p_value = 1.0  # In case all differences are zero or input is invalid

        adjusted_p = p_value * num_comparisons if correction == 'bonferroni' else p_value
        adjusted_p = min(adjusted_p, 1.0)

        results.append({
            'model_a': model_a,
            'model_b': model_b,
            'p_value': p_value,
            'adjusted_p': adjusted_p,
            'significant': adjusted_p < alpha
        })

    return results
