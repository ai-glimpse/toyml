import math

from typing import Dict


def normalize_log_probs(log_probs: Dict[int, float]) -> Dict[int, float]:
    """
    Normalize log probabilities using the log-sum-exp trick.

    Args:
        log_probs: Dictionary mapping from IDs to log probabilities

    Returns:
        Dictionary mapping from IDs to normalized probabilities (sums to 1.0)
    """
    # Find the maximum log probability to shift by (for numerical stability)
    max_log_prob = max(log_probs.values())

    # Calculate exp(log_prob - max_log_prob) for each probability
    exp_probs = {}
    exp_sum = 0.0

    for id_, log_prob in log_probs.items():
        # Subtract max_log_prob for numerical stability
        exp_prob = math.exp(log_prob - max_log_prob)
        exp_probs[id_] = exp_prob
        exp_sum += exp_prob
    print(exp_sum)
    # Normalize to get final probabilities
    normalized_probs = {id_: exp_prob / exp_sum for id_, exp_prob in exp_probs.items()}

    return normalized_probs


if __name__ == "__main__":
    # Example usage
    log_probs = {
        1: -980.3129721979374,
        2: -981.8851425503759,
        3: -911.9941418835292,
        4: -988.5867956213815,
        5: -997.0241252138998,
        6: -995.7041458755068,
    }

    normalized = normalize_log_probs(log_probs)

    # Print results with formatting
    print("Normalized probabilities:")
    for id_, prob in normalized.items():
        print(f"ID {id_}: {prob:.10f}")

    # Verify that probabilities sum to 1
    total = sum(normalized.values())
    print(f"\nSum of probabilities: {total:.10f}")
